"""
Starts a GUI for downloading files over HTTP.
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import functools
import logging
import mimetypes
import platform
import re
import threading
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path, PurePosixPath
from tkinter import Canvas, Event, StringVar, Tk, Widget
from tkinter.ttk import Button, Entry, Frame, Progressbar, Scrollbar, Style
from typing import Any, BinaryIO, Protocol, Self
from weakref import WeakSet

import httpx

log = logging.getLogger(__name__)


def _get_version() -> str:
    from importlib.metadata import PackageNotFoundError, version

    try:
        return version("parallel-downloader")
    except PackageNotFoundError:
        return ""


__version__ = _get_version()


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
    def __init__(self, event_thread: EventThread):
        super().__init__()

        self.event_thread = event_thread
        self._connect_lifetime_with_event_thread(event_thread)

        self.title("Parallel Downloader")
        self.geometry("600x600")

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self.frame = Frame(self)
        self.frame.grid()

        self.bind("<<Destroy>>", self._on_destroy)

    def switch_frame(self, frame: Frame) -> None:
        self.frame.destroy()
        self.frame = frame
        self.frame.grid(sticky="nesw")

    def _connect_lifetime_with_event_thread(self, event_thread: EventThread) -> None:
        # In our application we'll be running an asyncio event loop in
        # a separate thread. This event loop may try to run methods on
        # our GUI like event_generate(), which requires the GUI to be running.
        # If the GUI is destroyed first, it may cause a deadlock
        # in the other thread, preventing our program from exiting.
        # As such, we need to defer GUI destruction until the event thread
        # is finished.
        event_callback = lambda fut: self.event_generate("<<Destroy>>")
        event_thread.finished_fut.add_done_callback(event_callback)

    def destroy(self) -> None:
        self.event_thread.stop()

    def _on_destroy(self, event: Event) -> None:
        super().destroy()


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
        self.list.grid(row=0, column=0, sticky="nesw")
        self.total_progress = Progressbar(self)
        self.total_progress.grid(row=1, column=0, sticky="ew")
        self.controls = TkDownloadControls(self)
        self.controls.grid(row=2, column=0, sticky="es")

        self.refresh()

    def refresh(self) -> None:
        self._update_total_progress()
        if self.list.is_running():
            self.total_progress.state(["!disabled"])
        else:
            self.total_progress.state(["disabled"])

        self.list.refresh()
        self.controls.refresh()

    def _update_total_progress(self) -> None:
        if self.list.total > 0 or not self.list.is_running():
            self.total_progress["value"] = self.list.progress
            self.total_progress["maximum"] = self.list.total
            self.total_progress["mode"] = "determinate"
        else:
            steps = 200 * sum(e.is_running() for e in self.list.entries)
            self.total_progress["maximum"] = steps
            self.total_progress["mode"] = "indeterminate"
            self.total_progress.step()


# https://gist.github.com/thegamecracks/ee5614aa932c2167918a3c3dcc013710
class ScrollableFrame(Frame):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        self._canvas = Canvas(self, highlightthickness=0)
        self._canvas.grid(row=0, column=0, sticky="nesw")

        self._xscrollbar = Scrollbar(self, orient="horizontal")
        self._yscrollbar = Scrollbar(self, orient="vertical")

        # Use rowspan=2 or columnspan=2 appropriately if filling the
        # bottom-right corner of the frame is desired.
        self._xscrollbar.grid(row=1, column=0, sticky="ew")
        self._yscrollbar.grid(row=0, column=1, sticky="ns")

        self._xscrollbar["command"] = self._canvas.xview
        self._yscrollbar["command"] = self._canvas.yview
        self._canvas["xscrollcommand"] = self._wrap_scrollbar_set(self._xscrollbar)
        self._canvas["yscrollcommand"] = self._wrap_scrollbar_set(self._yscrollbar)

        self.inner = Frame(self._canvas)
        self._inner_id = self._canvas.create_window(
            (0, 0), window=self.inner, anchor="nw"
        )

        self._canvas.bind("<Configure>", lambda event: self._update())
        self.inner.bind("<Configure>", self._on_inner_configure)

        self._scrolled_widgets = WeakSet()
        self._style = Style(self)
        self._update_rate = 125
        self._update_loop()

    def _on_inner_configure(self, event: Event):
        background = self._style.lookup(self.inner.winfo_class(), "background")
        self._canvas.configure(background=background)
        self._update()

    def _update_loop(self):
        # Without this, any changes to the inner frame won't affect
        # the scroll bar/region until the window is resized.
        self._update()
        self.after(self._update_rate, self._update_loop)

    def _update(self):
        # self._canvas.bbox("all") doesn't update until window resize
        # so we're relying on the inner frame's requested height instead.
        new_width = max(self._canvas.winfo_width(), self.inner.winfo_reqwidth())
        new_height = max(self._canvas.winfo_height(), self.inner.winfo_reqheight())
        bbox = (0, 0, new_width, new_height)
        self._canvas.configure(scrollregion=bbox)
        self._canvas.itemconfigure(self._inner_id, width=new_width, height=new_height)

        self._update_scrollbar_visibility(self._xscrollbar)
        self._update_scrollbar_visibility(self._yscrollbar)
        self._propagate_scroll_binds(self.inner)

    def _propagate_scroll_binds(self, parent: Widget):
        if parent not in self._scrolled_widgets:
            parent.bind("<MouseWheel>", self._on_mouse_yscroll)
            parent.bind("<Shift-MouseWheel>", self._on_mouse_xscroll)
            self._scrolled_widgets.add(parent)

        for child in parent.winfo_children():
            self._propagate_scroll_binds(child)

    def _on_mouse_xscroll(self, event: Event):
        delta = int(-event.delta / 100)
        self._canvas.xview_scroll(delta, "units")

    def _on_mouse_yscroll(self, event: Event):
        delta = int(-event.delta / 100)
        self._canvas.yview_scroll(delta, "units")

    def _update_scrollbar_visibility(self, scrollbar: Scrollbar):
        if scrollbar.get() == (0, 1):
            scrollbar.grid_remove()
        else:
            scrollbar.grid()

    def _wrap_scrollbar_set(self, scrollbar: Scrollbar):
        def wrapper(*args, **kwargs):
            scrollbar.set(*args, **kwargs)
            self._update_scrollbar_visibility(scrollbar)

        return wrapper


class TkDownloadList(ScrollableFrame):
    def __init__(self, parent: TkDownloadFrame) -> None:
        super().__init__(parent)

        self.frame = parent

        self.inner.grid_columnconfigure(0, weight=1)

        self.entries: list[TkDownloadEntry] = []
        self.new_entry = Button(self.inner, text="+", command=self._do_add_entry)
        self.new_entry.grid(row=999, column=0, sticky="e")

    @property
    def progress(self) -> int:
        return sum(e.progress for e in self.entries if e.has_started())

    @property
    def total(self) -> int:
        return sum(e.total for e in self.entries if e.has_started())

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
        entry = TkDownloadEntry(self.inner, list_=self)
        self.add_entry(entry)
        self.frame.refresh()


class TkDownloadEntry(Frame):
    download: Download | None
    _file: BinaryIO | None

    def __init__(self, parent: Frame, list_: TkDownloadList) -> None:
        super().__init__(parent)

        self.list = list_

        self.grid_columnconfigure(0, weight=1)

        self.url_entry_var = StringVar(self)
        self.url_entry_var.trace_add("write", lambda *args: self._check_url())
        self.url_entry = Entry(self, textvariable=self.url_entry_var)
        self.url_entry_valid = not self.list.frame.validate_urls

        self.status_entry_var = StringVar(self)
        self.status_entry = Entry(self, textvariable=self.status_entry_var)
        self.status_entry.state(["disabled"])

        self.progress_bar = Progressbar(self)

        for widget in (self.progress_bar, self.status_entry, self.url_entry):
            widget.grid(row=0, column=0, sticky="ew")

        self.remove = Button(self, text="X", command=self._do_remove)
        self.remove.grid(row=0, column=1)

        self._reset()
        self.refresh()

    def _reset(self) -> None:
        self.download = None
        self.filename = ""
        self.progress = 0
        self.total = 0
        self._file = None
        self._interrupted = False
        self._set_status("")

    def start(self) -> None:
        # NOTE: for now, a download entry can only be started once
        if not self.can_start():
            return

        self.cancel()
        self._reset()

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

        self._set_status(f"Interrupted download for {self.filename}")
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
            self.url_entry.grid_remove()
            self.status_entry.grid_remove()
            self.progress_bar.grid()

            self._update_progress()
            self.remove.state(["disabled"])
        elif self.status_entry.get() != "":
            self.url_entry.grid_remove()
            self.progress_bar.grid_remove()
            self.status_entry.grid()

            self.remove.state(["!disabled"])
        elif self.has_started():
            self.status_entry.grid_remove()
            self.progress_bar.grid_remove()
            self.url_entry.grid()

            self.url_entry.state(["disabled"])
            self.remove.state(["!disabled"])
        else:
            self.status_entry.grid_remove()
            self.progress_bar.grid_remove()
            self.url_entry.grid()

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

    def _update_progress(self) -> None:
        if self.total > 0 or not self.is_running():
            self.progress_bar["value"] = self.progress
            self.progress_bar["maximum"] = self.total
            self.progress_bar["mode"] = "determinate"
        else:
            self.progress_bar["maximum"] = 200
            self.progress_bar["mode"] = "indeterminate"
            self.progress_bar.step()

    def _download_callback(self, chunk: DownloadChunk | BaseException | None) -> None:
        # NOTE: this callback runs in the event loop's thread
        if self._interrupted:
            return

        assert self.download is not None

        if chunk is None or isinstance(chunk, BaseException):
            return self._end_download(chunk)

        file = self._try_open_file(chunk.filename)
        if file is None:
            return self.cancel()

        self.filename = chunk.filename
        self.progress = chunk.progress
        self.total = chunk.total

        file.write(chunk.data)
        self.list.frame.refresh()  # FIXME: refresh independently of download?

    def _try_open_file(self, filename: str) -> BinaryIO | None:
        if self._file is not None:
            return self._file
        elif self._interrupted:
            return

        try:
            self._file = self.list.frame.writer_factory(filename)
        except FileExistsError:
            log.error("File %r already exists", filename)
            self._set_status(f"File {filename} already exists")
            return
        except Exception:
            log.exception("Failed to open %s", filename)
            self._set_status(f"Failed to open {filename}")
            return
        else:
            return self._file

    def _end_download(self, exc: BaseException | None) -> None:
        filename = self.filename or self.url_entry.get()
        if exc is not None:
            log.error("Failed to download %r", filename)
            self._set_status(f"Failed to download {filename}")
        elif self.filename == "":
            log.error("No data received for %r", filename)
            self._set_status("No data received")
        else:
            log.info("Successfully downloaded %r", filename)
            self._set_status(f"Successfully downloaded {filename}")

        self._close_file()
        self.list.frame.refresh()

    def _close_file(self) -> None:
        self._interrupted = True
        if self._file is not None:
            self._file.close()
            self._file = None

    def _set_status(self, message: str) -> None:
        if message == "" or self.status_entry_var.get() == "":
            self.status_entry_var.set(message)

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
    """A controllable, one-time download.

    All methods provided should be idempotent.

    """

    def start(self) -> None:
        """Starts the download.

        This should cause is_running() to immediately return True
        unless the download was already cancelled.

        """
        ...

    def pause(self) -> None:
        """Pauses the download.

        If the download has not yet started, the download should be
        paused as soon as it starts.

        """
        ...

    def resume(self) -> None:
        """Resumes the download.

        If the download has not yet started, the download should be
        resumed as soon as it starts.

        """
        ...

    def cancel(self) -> None:
        """Cancels the download.

        This should cause is_running() to immediately return False,
        even if the underlying download has not yet been cancelled.

        If the download is not yet started, this should prevent the
        download from running.

        """
        ...

    def is_running(self) -> bool:
        """Checks if the download is currently in progress.

        This should return True as soon as start() is called, unless
        cancel() was called beforehand. It should also return True when
        the download is paused.

        """
        ...

    def is_paused(self) -> bool:
        """Checks if the download is currently paused.

        This can be True or False regardless of whether the download is
        running.

        """
        ...


class DownloadCallback(Protocol):
    def __call__(self, chunk: DownloadChunk | BaseException | None, /) -> Any: ...


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
        self._started = False
        self._stopped = False
        self._paused = False
        self._interval_ms = 40

    def start(self) -> None:
        if self._started:
            return

        self._started = True
        self._step()

    def pause(self) -> None:
        self._paused = True

    def resume(self) -> None:
        self._paused = False

    def cancel(self) -> None:
        self._stopped = True

    def is_running(self) -> bool:
        return not self._stopped and self._progress < self._total

    def is_paused(self) -> bool:
        return self._paused

    def _step(self) -> None:
        if self._stopped:
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
        self._fut.add_done_callback(self._send_end_of_file)

    def pause(self) -> None:
        self.thread.loop.call_soon_threadsafe(self._resume.clear)

    def resume(self) -> None:
        self.thread.loop.call_soon_threadsafe(self._resume.set)

    def cancel(self) -> None:
        if self._fut is not None:
            self._fut.cancel()

    def is_running(self) -> bool:
        return self._fut is not None and not self._fut.done()

    def is_paused(self) -> bool:
        return not self._resume.is_set()

    async def _start_stream(self) -> None:
        await self._resume.wait()

        async with self.client.stream(
            "GET",
            self.url,
            follow_redirects=True,
        ) as response:
            response.raise_for_status()
            await self._download_stream(response)

    async def _download_stream(self, response: httpx.Response) -> None:
        content_type: str = response.headers.get("Content-Type", "")
        filename = self.get_filename_from_url(self.url, content_type)

        log.debug("Response received for %r", self.url)

        total = int(response.headers.get("Content-Length", "0"))
        log.debug("Expecting %d bytes for %r", total, self.url)

        await self._resume.wait()

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

    def _send_end_of_file(self, fut: concurrent.futures.Future) -> None:
        if fut.cancelled():
            self.callback(None)
        else:
            self.callback(fut.exception())

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
        self.join()

    @property
    def loop(self) -> asyncio.AbstractEventLoop:
        return self.loop_fut.result()

    def run(self) -> None:
        try:
            asyncio.run(self._run_forever())
        finally:
            self.finished_fut.set_result(None)

    def stop(self) -> None:
        if not self.stop_fut.done():
            self.stop_fut.set_result(None)

    async def _run_forever(self) -> None:
        self.loop_fut.set_result(asyncio.get_running_loop())
        await asyncio.wrap_future(self.stop_fut)


def configure_logging(verbose: int) -> None:
    if verbose == 0:
        fmt = "%(levelname)s: %(message)s"
        level = logging.WARNING
    elif verbose == 1:
        fmt = "%(levelname)s: %(message)s"
        level = logging.INFO
    else:
        fmt = "%(levelname)s: %(message)-50s (%(name)s#L%(lineno)d)"
        level = logging.DEBUG

    logging.basicConfig(format=fmt, level=level)


def enable_windows_dpi_awareness():
    if platform.system() == "Windows":
        from ctypes import windll

        windll.shcore.SetProcessDpiAwareness(2)


def main():
    parser = argparse.ArgumentParser(
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
        "-V",
        "--version",
        action="version",
        version=f"{parser.prog} {__version__}",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument(
        "-n",
        "--dry-run",
        action="store_true",
        help="Don't actually download anything",
    )
    output_group.add_argument(
        "-o",
        "--output-dir",
        default=Path(),
        type=Path,
        help="The directory to write files to (defaults to CWD)",
    )

    args = parser.parse_args()
    verbose: int = args.verbose
    dry_run: bool = args.dry_run
    output_dir: Path = args.output_dir

    configure_logging(verbose)
    enable_windows_dpi_awareness()

    with EventThread() as event_thread:
        app = TkApp(event_thread)
        if dry_run:
            download_factory = functools.partial(DummyDownload, app)
            writer_factory = lambda *_: BytesIO()
        else:
            download_factory = functools.partial(
                HTTPXDownload,
                event_thread,
                httpx.AsyncClient(),  # TODO: use with context manager
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            writer_factory = lambda filename: open(
                output_dir / filename,
                mode="xb",
            )

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
