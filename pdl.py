from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import mimetypes
import re
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import PurePosixPath
from tkinter import StringVar, Tk
from tkinter.ttk import Button, Entry, Frame, Progressbar
from typing import Any, BinaryIO, Protocol, Self

import httpx

log = logging.getLogger(__name__)

# ===== UI =====

HTTP_URL_REGEX = re.compile(
    # https://urlregex.com/
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+",
)


def is_valid_url(url: str) -> bool:
    return HTTP_URL_REGEX.fullmatch(url) is not None


def log_fut_exception(fut: concurrent.futures.Future) -> None:
    if fut.cancelled():
        return

    exc = fut.exception()
    if exc is not None:
        log.error("Uncaught exception from future %r", fut, exc_info=exc)


class TkApp(Tk):
    def __init__(self):
        super().__init__()

        self.title("Parallel Downloader")
        self.geometry("480x480")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame = Frame(self)
        self.frame.grid()

    def switch_frame(self, frame: Frame) -> None:
        self.frame.destroy()
        self.frame = frame
        self.frame.grid(sticky="nesw")


class TkDownloadFrame(Frame):
    def __init__(
        self,
        app: TkApp,
        *,
        download_factory: DownloadFactory,
        writer_factory: WriterFactory,
        validate_urls: bool = True,
    ) -> None:
        super().__init__(app, padding=10)

        self.app = app
        self.download_factory = download_factory
        self.writer_factory = writer_factory
        self.validate_urls = validate_urls

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.list = TkDownloadList(self)
        self.list.grid(row=0, column=0, sticky="new")
        self.total_progress = Progressbar(self)
        self.total_progress.grid(row=1, column=0, sticky="ew")
        self.controls = TkDownloadControls(self)
        self.controls.grid(row=2, column=0, sticky="es")

        self.refresh()

    def refresh(self) -> None:
        if self.list.is_running():
            self.total_progress["value"] = self.list.progress
            self.total_progress["maximum"] = self.list.total
            self.total_progress["mode"] = (
                "determinate" if self.list.total > 0 else "indeterminate"
            )
            self.total_progress.state(["!disabled"])
        else:
            self.total_progress.state(["disabled"])

        self.list.refresh()
        self.controls.refresh()


class TkDownloadList(Frame):
    def __init__(self, parent: TkDownloadFrame) -> None:
        super().__init__(parent)

        self.frame = parent

        self.grid_columnconfigure(0, weight=1)

        self.entries: list[TkDownloadEntry] = []
        self.new_entry = Button(self, text="+", command=self._do_add_entry)
        self.new_entry.grid(row=999, column=0, sticky="e")

    @property
    def progress(self) -> int:
        return sum(e.progress for e in self.entries if e.is_running())

    @property
    def total(self) -> int:
        return sum(e.total for e in self.entries if e.is_running())

    def is_running(self) -> bool:
        return any(e.is_running() for e in self.entries)

    def is_paused(self) -> bool:
        return any(e.is_paused() for e in self.entries)

    def can_start(self) -> bool:
        return any(e.can_start() for e in self.entries)

    def add_entry(self, entry: TkDownloadEntry) -> None:
        entry.grid(row=len(self.entries), column=0, sticky="ew")
        self.entries.append(entry)

    def remove_entry(self, entry: TkDownloadEntry) -> None:
        i = self.entries.index(entry)
        del self.entries[i]
        entry.destroy()

        # Re-grid existing entries
        entries = self.entries.copy()
        self.entries.clear()

        for entry in entries:
            self.add_entry(entry)

    def start_all(self) -> None:
        for entry in self.entries:
            entry.start()

    def pause_all(self) -> None:
        for entry in self.entries:
            entry.pause()

    def resume_all(self) -> None:
        for entry in self.entries:
            entry.resume()

    def cancel_all(self) -> None:
        for entry in self.entries:
            entry.cancel()

    def refresh(self) -> None:
        if self.is_running():
            self.new_entry.state(["disabled"])
        else:
            self.new_entry.state(["!disabled"])

        for entry in self.entries:
            entry.refresh()

    def _do_add_entry(self) -> None:
        entry = TkDownloadEntry(self)
        self.add_entry(entry)
        self.frame.refresh()


class TkDownloadEntry(Frame):
    def __init__(self, parent: TkDownloadList) -> None:
        super().__init__(parent)

        self.list = parent

        self.download: Download | None = None
        self.filename = ""
        self.progress = 0
        self.total = 0
        self._file: BinaryIO | None = None
        self._interrupted = False

        self.grid_columnconfigure(0, weight=1)

        self.url_entry_var = StringVar(self)
        self.url_entry_var.trace_add("write", lambda *args: self._check_url())
        self.url_entry = Entry(self, textvariable=self.url_entry_var)
        self.url_entry_valid = not self.list.frame.validate_urls

        self.progress_bar = Progressbar(self)

        self.remove = Button(self, text="X", command=self._do_remove)
        self.remove.grid(row=0, column=1)

        self.refresh()

    def start(self) -> None:
        # NOTE: for now, a download entry can only be started once
        if not self.can_start():
            return

        self.cancel()

        # FIXME: it's not obvious what attributes need to be reset
        self.filename = ""
        self._interrupted = False
        url = self.url_entry.get()
        log.debug("Starting download for url %r", url)

        self.download = self.list.frame.download_factory(
            url,
            self._download_callback,
        )
        self.download.start()

    def pause(self) -> None:
        if self.download is None:
            return

        self.download.pause()

    def resume(self) -> None:
        if self.download is None:
            return

        self.download.resume()

    def cancel(self) -> None:
        if self.download is None:
            return

        self._close_file()
        self.download.cancel()
        self.list.frame.refresh()

    def is_running(self) -> bool:
        return (
            self.download is not None
            and self.download.is_running()
            and not self._interrupted  # should be set upon cancellation
        )

    def is_paused(self) -> bool:
        return (
            self.download is not None
            and self.is_running()
            and self.download.is_paused()
        )

    def has_started(self) -> bool:
        return self.download is not None

    def can_start(self) -> bool:
        return not self.has_started() and self.url_entry_valid

    def refresh(self) -> None:
        if self.is_running():
            self.progress_bar["value"] = self.progress
            self.progress_bar["maximum"] = self.total
            self.progress_bar["mode"] = (
                "determinate" if self.total > 0 else "indeterminate"
            )
            self.url_entry.grid_forget()
            self.progress_bar.grid(row=0, column=0, sticky="ew")
            self.remove.state(["disabled"])
        elif self.has_started():
            self.progress_bar.grid_forget()
            self.url_entry.grid(row=0, column=0, sticky="ew")
            self.url_entry.state(["disabled"])
            self.remove.state(["!disabled"])
        else:
            self.progress_bar.grid_forget()
            self.url_entry.grid(row=0, column=0, sticky="ew")
            self.url_entry.state(["!disabled"])
            self.remove.state(["!disabled"])

    def _do_remove(self) -> None:
        self.list.remove_entry(self)
        self.list.frame.refresh()

    def _check_url(self) -> None:
        if self.list.frame.validate_urls:
            self.url_entry_valid = is_valid_url(self.url_entry.get())
        else:
            self.url_entry_valid = True
        self.list.frame.refresh()

    def _download_callback(self, chunk: DownloadChunk | None) -> None:
        if self._interrupted:
            return

        assert self.download is not None

        if chunk is None:
            return self._end_download()

        self.filename = chunk.filename
        self.progress = chunk.progress
        self.total = chunk.total

        file = self._try_open_file(chunk.filename)
        if file is not None:
            file.write(chunk.data)
            self.list.frame.refresh()  # FIXME: refresh independently of download?
        else:
            self.cancel()

    def _try_open_file(self, filename: str) -> BinaryIO | None:
        if self._file is not None:
            return self._file

        try:
            self._file = self.list.frame.writer_factory(filename)
        except Exception:
            # TODO: show download failure to user
            log.exception("Failed to open %s", filename)
            return
        else:
            return self._file

    def _end_download(self) -> None:
        if self.filename == "":
            log.error("No data received for %r", self.url_entry.get())
        else:
            log.info("Successfully downloaded %s", self.filename)

        self._close_file()
        self.list.frame.refresh()

    def _close_file(self) -> None:
        if self._file is None:
            return

        self._file.close()
        self._file = None
        self._interrupted = True

    def destroy(self) -> None:
        self._close_file()
        return super().destroy()


class TkDownloadControls(Frame):
    def __init__(self, parent: TkDownloadFrame) -> None:
        super().__init__(parent)

        self.frame = parent

        self.stop = Button(self, text="Stop", command=self._do_stop)
        self.stop.grid(row=0, column=0)
        self.pause = Button(self)
        self.pause.grid(row=0, column=1)
        self.start = Button(self, text="Start", command=self._do_start)
        self.start.grid(row=0, column=2)

        self.refresh()

    def refresh(self) -> None:
        if self.frame.list.can_start():
            self.stop.state(["disabled"])
            self.pause.state(["disabled"])
            self.start.state(["!disabled"])
        elif self.frame.list.is_running():
            self.stop.state(["!disabled"])
            self.pause.state(["!disabled"])
            self.start.state(["disabled"])
        else:
            self.stop.state(["disabled"])
            self.pause.state(["disabled"])
            self.start.state(["disabled"])

        if self.frame.list.is_paused():
            self.pause["text"] = "Resume"
            self.pause["command"] = self._do_resume
        else:
            self.pause["text"] = "Pause"
            self.pause["command"] = self._do_pause

    def _do_stop(self) -> None:
        self.frame.list.cancel_all()
        self.frame.refresh()

    def _do_resume(self) -> None:
        self.frame.list.resume_all()
        self.frame.refresh()

    def _do_pause(self) -> None:
        self.frame.list.pause_all()
        self.frame.refresh()

    def _do_start(self) -> None:
        self.frame.list.start_all()
        self.frame.refresh()


# ===== IO =====


class WriterFactory(Protocol):
    def __call__(self, filename: str, /) -> BinaryIO: ...


class DownloadFactory(Protocol):
    def __call__(self, url: str, callback: DownloadCallback, /) -> Download: ...


class Download(Protocol):
    def start(self) -> None: ...
    def pause(self) -> None: ...
    def resume(self) -> None: ...
    def cancel(self) -> None: ...
    def is_running(self) -> bool: ...
    def is_paused(self) -> bool: ...


class DownloadCallback(Protocol):
    def __call__(self, chunk: DownloadChunk | None, /) -> Any: ...


@dataclass(kw_only=True)
class DownloadChunk:
    filename: str
    data: bytes = field(repr=False)
    progress: int
    total: int

    def __str__(self) -> str:
        return "{} bytes for {}, {}/{} ({:.0%})".format(
            len(self.data),
            self.filename,
            self.progress,
            self.total,
            self.progress / self.total if self.total > 0 else float("inf"),
        )


class DummyDownload(Download):
    def __init__(self, app: TkApp, url: str, callback: DownloadCallback) -> None:
        self.app = app
        self.url = url
        self.callback = callback

        self._data = b"foobar\n"
        self._progress = 0
        self._total = 100
        self._stop = False
        self._paused = False
        self._interval_ms = 40

    def start(self) -> None:
        self._step()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def cancel(self) -> None:
        self._stop = True

    def is_running(self) -> bool:
        return not self._stop and self._progress < self._total

    def is_paused(self) -> bool:
        return self._paused

    def _step(self) -> None:
        if self._stop:
            return
        elif self._paused:
            self.app.after(self._interval_ms, self._step)
            return

        self._progress += 1
        self.callback(
            DownloadChunk(
                filename="foobar.txt",
                data=self._data,
                progress=self._progress,
                total=self._total,
            )
        )

        if self._progress >= self._total:
            return self.callback(None)

        self.app.after(self._interval_ms, self._step)


class HTTPXDownload(Download):
    _fut: concurrent.futures.Future | None

    def __init__(
        self,
        thread: EventThread,
        client: httpx.AsyncClient,
        url: str,
        callback: DownloadCallback,
    ) -> None:
        self.thread = thread
        self.client = client
        self.url = url
        self.callback = callback

        self._fut = None
        self._resume = asyncio.Event()
        self._resume.set()

    def start(self) -> None:
        if self._fut is not None:
            return

        self._fut = asyncio.run_coroutine_threadsafe(
            self._start_stream(),
            self.thread.loop,
        )
        self._fut.add_done_callback(log_fut_exception)
        self._fut.add_done_callback(lambda fut: self.callback(None))

    def pause(self) -> None:
        self.thread.loop.call_soon_threadsafe(self._resume.clear)

    def resume(self) -> None:
        self.thread.loop.call_soon_threadsafe(self._resume.set)

    def cancel(self) -> None:
        if self._fut is None or self._fut.cancelled():
            return

        self._fut.cancel()

    def is_running(self) -> bool:
        return self._fut is not None and not self._fut.done()

    def is_paused(self) -> bool:
        return not self._resume.is_set()

    async def _start_stream(self) -> None:
        async with self.client.stream(
            "GET",
            self.url,
            follow_redirects=True,
        ) as response:
            await self._download_stream(response)

    async def _download_stream(self, response: httpx.Response) -> None:
        content_type: str = response.headers.get("Content-Type", "")
        filename = self.get_filename_from_url(self.url, content_type)

        log.debug("Response received for %r", self.url)

        total = int(response.headers.get("Content-Length", "0"))
        log.debug("Expecting %d bytes for %r", total, self.url)

        async for data in response.aiter_bytes():
            chunk = DownloadChunk(
                filename=filename,
                data=data,
                progress=response.num_bytes_downloaded,
                total=total,
            )
            log.debug("Received chunk %s", chunk)
            self.callback(chunk)

            await self._resume.wait()

    @staticmethod
    def get_filename_from_url(raw_url: str, content_type: str) -> str:
        content_type, *_ = content_type.split(";", 1)

        url = httpx.URL(raw_url)
        path = PurePosixPath(url.path)
        if path.stem == "":
            return url.host + ".html"
        elif path.suffix == "" and content_type == "text/plain":
            return path.stem + ".html"
        elif path.suffix == "":
            ext = mimetypes.guess_extension(content_type)
            if ext is not None:
                return path.name + ext

        return path.name


class EventThread(threading.Thread):
    """Handles starting and stopping an asyncio event loop in a separate thread."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.loop_fut = concurrent.futures.Future()
        self.stop_fut = concurrent.futures.Future()
        self.finished_fut = concurrent.futures.Future()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, tb) -> None:
        self.stop()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self.loop_fut.result()

    def run(self) -> None:
        try:
            asyncio.run(self._run_forever())
        finally:
            self.finished_fut.set_result(None)

    def stop(self) -> None:
        self.stop_fut.set_result(None)
        self.finished_fut.result()

    async def _run_forever(self) -> None:
        self.loop_fut.set_result(asyncio.get_running_loop())
        await asyncio.wrap_future(self.stop_fut)


def configure_logging(verbose: int) -> None:
    if verbose == 0:
        return
    elif verbose == 1:
        fmt = "%(levelname)s: %(message)s"
        level = logging.INFO
    else:
        fmt = "%(levelname)s: %(message)-50s (%(name)s#L%(lineno)d)"
        level = logging.DEBUG

    logging.basicConfig(format=fmt, level=level)


def main():
    parser = argparse.ArgumentParser(
        prog=__package__,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase logging verbosity",
    )
    parser.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Don't actually download anything",
    )

    args = parser.parse_args()

    configure_logging(args.verbose)

    app = TkApp()
    with EventThread() as event_thread:
        if args.dry_run:
            download_factory = functools.partial(DummyDownload, app)
            writer_factory = lambda *_: BytesIO()
        else:
            download_factory = functools.partial(
                HTTPXDownload,
                event_thread,
                httpx.AsyncClient(),  # TODO: use with context manager
            )
            writer_factory = functools.partial(open, mode="xb")

        app.switch_frame(
            TkDownloadFrame(
                app,
                download_factory=download_factory,
                writer_factory=writer_factory,
                validate_urls=not args.dry_run,
            )
        )
        app.mainloop()


if __name__ == "__main__":
    main()
