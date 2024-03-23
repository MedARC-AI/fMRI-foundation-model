import numpy as np
from torch import Tensor, stack
import lightning as pl
from torch.utils.data import IterDataPipe, DataLoader
from torch.utils.data.datapipes._decorator import functional_datapipe
from torch.utils.data.datapipes.iter.callable import MapperIterDataPipe
from torch.utils.data.datapipes.utils.common import StreamWrapper, match_masks
from torchdata.datapipes.iter import IterableWrapper, IterDataPipe, TarArchiveLoader
from torchdata.datapipes.iter.load.s3io import S3FileLoaderIterDataPipe
from torchdata.datapipes.utils.common import validate_pathname_binary_tuple
import warnings
import subprocess
import tarfile
import tempfile
import time
import random
from omegaconf import DictConfig, ListConfig
import webdataset as wds
import re
import os
import copy
from io import BufferedIOBase, BytesIO, RawIOBase
from typing import (
    IO,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
)
T_co = TypeVar("T_co", covariant=True)

class ShuffledListDataPipe(IterDataPipe[T_co]):
    def __init__(
        self,
        source_list: List[T_co],
        *,
        shuffle: bool = True,
        cycle: Union[bool, int] = True,
    ):
        super().__init__()
        self.source = source_list
        self._enabled = shuffle
        self._seed = None
        self._rng = random.Random()
        if isinstance(cycle, bool):
            if cycle:
                self._cycle = -1
            else:
                self._cycle = 1
        else:
            assert isinstance(cycle, int)
            self._cycle = cycle

    def __iter__(self) -> Iterator[T_co]:
        source = copy.copy(self.source)
        cycle = self._cycle
        epochs = 0
        while cycle == -1 or epochs < cycle:
            if self._enabled:
                self._rng.shuffle(source)
            yield from source
            epochs += 1

    def __len__(self):
        if self.count == -1:
            raise TypeError(
                f"This {type(self).__name__} instance cycles forever, and "
                f"therefore doesn't have valid length."
            )
        else:
            return self.count * len(self.source)

    def __getstate__(self):
        state = (
            self.source,
            self._enabled,
            self._seed,
            self._rng.getstate(),
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        )
        if IterDataPipe.getstate_hook is not None:
            return IterDataPipe.getstate_hook(state)
        return state

    def __setstate__(self, state):
        (
            self.source,
            self._enabled,
            self._seed,
            rng_state,
            self._valid_iterator_id,
            self._number_of_samples_yielded,
        ) = state
        self._rng = random.Random()
        self._rng.setstate(rng_state)

def get_job_temp_dir(dl_root: str) -> str:
    try:
        job_or_array_id = (
            os.environ.get("SLURM_ARRAY_JOB_ID", "") or os.environ["SLURM_JOB_ID"]
        )
        task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "") or "0"
        return os.path.join(dl_root, f"JOB_{job_or_array_id}/TASK_{task_id}")
    except KeyError as e:
        print("SLURM_JOB_ID env var not set: You sure this job is healthy?")
        raise e

def _shard_expand(s: str) -> List[str]:
    expansion = r"\{[0-9]+\.\.[0-9]+\}"
    m = re.search(expansion, s)
    if not m:
        return [s]
    prefix = s[: m.start()]
    rest = _shard_expand(s[m.end() :])
    rng = s[m.start() + 1 : m.end() - 1]
    lohi = rng.split("..")
    if len(lohi[0]) == len(lohi[1]) and lohi[0].startswith("0"):
        fmt = "{prefix}{i:0>{l}d}{r}"
    elif len(lohi[0]) <= len(lohi[1]):
        if lohi[0].startswith("0") and lohi[0] != "0":
            raise ValueError(
                "shard_expand: low bound must not start with 0 if low bound is shorter"
            )
        fmt = "{prefix}{i}{r}"
    else:
        raise ValueError("shard_expand: low bound must be shorter than high bound")
    lo, hi = (int(x) for x in lohi)
    if lo >= hi:
        raise ValueError(f"shard_expand: bad range in in shard spec {s}.")
    result = []
    for i in range(lo, hi + 1):
        for r in rest:
            expanded: str = fmt.format(prefix=prefix, i=i, r=r, l=len(lohi[1]))
            result.append(expanded)
    return result

@functional_datapipe("custom_shard_expand")
class CustomShardExpanderIterDataPipe(IterDataPipe[str]):
    def __init__(self, source_datapipe: IterDataPipe[str]) -> None:
        super().__init__()
        self.source_datapipe: IterDataPipe[str] = source_datapipe

    def __iter__(self) -> Iterator[str]:
        for path in self.source_datapipe:
            yield from _shard_expand(path)

def warn_and_continue(exn):
    """Call in an exception handler to ignore any exception, issue a warning, and continue."""
    print(exn)
    warnings.warn(repr(exn))
    time.sleep(0.05)
    return True
error_handler = warn_and_continue

def is_stream_handle(data):
    obj_to_check = data.file_obj if isinstance(data, StreamWrapper) else data
    return isinstance(obj_to_check, (BufferedIOBase, RawIOBase))

def read_stream_handles(data):
    if not is_stream_handle(data):
        return data
    else:
        ds = data
        data = b"".join(data)
        ds.close()
        del ds
        return data
        
def stream_reader(sample):
    return {k: read_stream_handles(v) for k, v in sample.items()}

def to_our_format(sample):
    wds_key = sample.pop("__key__")
    sample = {k.lstrip("."): v for k, v in sample.items()}
    sample["__key__"] = wds_key.split("/")[-1]
    sample["__url__"] = "/".join(wds_key.split("/")[:-1])
    return sample

def add_processors(
    datapipeline,
    processors: Optional[ListConfig],
    description: str,
    error_handler: Callable = warn_and_continue,
):
    if not processors:
        return datapipeline
    else:
        for i, processor_config in enumerate(processors):
            processor = instantiate(processor_config)
            if isinstance(processor, AbstractFilter):
                print(
                    f"Adding filter {processor.__class__.__name__} as {description} #{i} "
                    f"to the datapipeline"
                )
                datapipeline = datapipeline.filter(processor.filter)
            elif isinstance(processor, AbstractMapper):
                print(
                    f"Adding mapper {processor.__class__.__name__} as {description} #{i} "
                    f"to the datapipeline"
                )
                datapipeline = datapipeline.map_with_handler(
                    processor.map,
                    handler=error_handler,
                    called_cls_name=processor.__class__.__name__,
                )
            else:
                raise TypeError(
                    f"chosen {description} {processor.__class__.__name__} should be either subclass"
                    "AbstractMapper or AbstractFilter"
                )
        return datapipeline

@functional_datapipe("map_with_handler")
class MapperWithErrorHandlingIterDataPipe(MapperIterDataPipe):
    def __init__(
        self,
        datapipe: IterDataPipe,
        fn: Callable,
        handler: Callable = wds.reraise_exception,
        input_col: Optional[Union[str, int]] = None,
        output_col: Optional[Union[str, int]] = None,
        called_cls_name: Optional[str] = None,
    ):
        # for now, disbable input and output col since this is never used anyways
        if input_col is not None:
            raise NotImplementedError("`input_col` argument currently not supported")

        if output_col is not None:
            raise NotImplementedError("`output_col` argument currently not supported")

        super().__init__(datapipe, fn)
        self.handler = handler
        self._apply_fn_ = self._apply_fn

    def __iter__(self) -> Iterator[T_co]:
        for data in self.datapipe:
            try:
                res = self._apply_fn_(data)
                if res is None:
                    continue
                yield res
            except Exception as e:
                if self.handler(e):
                    continue
                else:
                    raise e

__S3_TOOLS__ = {
    "s3": ["/usr/local/bin/aws", "s3"],
}

def is_tar(x: str) -> bool:
    return x.endswith(".tar")

def ls_aws(
    path: str,
    tool: str = "s3",
    recursive: bool = True,
    raise_errors: bool = True,
    skip_files: bool = True,
):
    assert path.startswith("s3://"), path

    # in case we have a file, set recursive to false
    isfile = bool(os.path.splitext(path)[1])
    if isfile and skip_files:
        return [path]

    if not path.endswith("/") and not isfile:
        path = path + "/"
    cmd = [*__S3_TOOLS__[tool], "ls", f"{path}"]

    if recursive and not isfile:
        cmd += ["--recursive"]
    try:
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        result.check_returncode()

        stdout = result.stdout.decode().split("\n")
        # return empty list in case path is not a real file/directory
        if not stdout[0]:
            return []

        # return path if it is an exisiting file
        if isfile:
            return [path]

        # strip timestamp and object size
        out = [line.split(" ")[-1] for line in stdout if line]
        if recursive:
            # aws s3 ls returns prefix+filename
            bucket = path[: path.find("/", len("s3://"))]  # s3://<bucket>/...
            out = [os.path.join(bucket, o) for o in out]
        else:
            # aws s3 ls returns filename only
            out = [os.path.join(path, o) for i in out]

        return out
    except subprocess.CalledProcessError as e:
        print(f"Got exception while trying to load data! {e.__class__.__name__}: {e}")
        if raise_errors:
            raise e
        else:
            return []

@functional_datapipe("wrapped_load_files_by_s3")
class WrappedS3FileLoaderIterDataPipe(S3FileLoaderIterDataPipe):
    def __init__(
        self,
        *args,
        ignore_missing_files: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.ignore_missing_files = ignore_missing_files

    def __iter__(self) -> Iterator[Tuple[str, StreamWrapper]]:
        # same as parent but maybe ignoring errors
        for url in self.source_datapipe:
            try:
                yield url, StreamWrapper(BytesIO(self.handler.s3_read(url)))
            except ValueError as e:
                if not self.ignore_missing_files:
                    raise FileNotFoundError(url)
                else:
                    print(f"Warning: Could not download {url}")

@functional_datapipe("list_files_with_aws_cli")
class S3CLIFileListerIterDataPipe(IterDataPipe[T_co]):
    def __init__(
        self,
        source_datapipe: Union[str, Sequence[str], IterDataPipe],
        s3_tool: str = "s3",
        n_retries: int = 10,
        sleep_interval: float = 0.01,
    ):
        if isinstance(source_datapipe, str):
            source_datapipe = [
                source_datapipe,
            ]
        if not isinstance(source_datapipe, IterDataPipe):
            self.datapipe: IterDataPipe = IterableWrapper(source_datapipe)  # type: ignore[assignment]
        else:
            self.datapipe = source_datapipe

        assert s3_tool in __S3_TOOLS__, f"`s3_tool` has to be in {list(__S3_TOOLS__)}"

        self.s3_tool = s3_tool

        self.n_retries = n_retries
        self.sleep = sleep_interval

    def __iter__(self) -> Iterator[str]:
        for root in self.datapipe:
            files = None
            for _ in range(self.n_retries):
                try:
                    files = ls_aws(
                        path=root, tool=self.s3_tool, recursive=True, raise_errors=True
                    )
                    break
                except subprocess.CalledProcessError:
                    time.sleep(self.sleep)

            if files is None:
                print(
                    f"Could not ls data expected under {root} in {self.n_retries} tries. Not yielding ..."
                )
                continue

            for file in files:
                yield file

def _download_tar(url, scratch, verbose=False, n_retries=100):
    start = time.perf_counter()
    if verbose:
        _log(f"downloading {url}")
    uid = "".join(filter(str.isalnum, os.path.splitext(url)[0]))
    idx = 0
    path = os.path.join(scratch, f"{os.getpid()}.{uid}.{idx}.tar")
    while os.path.exists(path):
        idx += 1
        path = os.path.join(scratch, f"{os.getpid()}.{uid}.{idx}.tar")

    os.makedirs(os.path.dirname(path), exist_ok=True)
    for i_try in range(n_retries):
        p = subprocess.call(
            [
                "/usr/local/bin/aws",
                "s3",
                "cp",
                url,
                path,
                "--quiet",
            ]
        )

        if p != 0:
            # mostly to work around
            # Error when retrieving credentials from Ec2InstanceMetadata: No
            # credentials found in credential_source referenced in profile
            # default
            if i_try == 0:
                # check if the requested file actually exists
                if not ls_aws(url, skip_files=False, raise_errors=False):
                    raise FileNotFoundError(url)

            if i_try + 1 < n_retries:
                time.sleep(random.uniform(0.0, 0.1))
                continue
            else:
                raise Exception(
                    f"File {url} should exist but failed to download after trying {i_try+1} times."
                )
        break

    if verbose:
        _log(
            f"wrote {path} in {time.perf_counter() - start} secs after {i_try+1} tries."
        )
    return path


@functional_datapipe("download_with_s3_cli")
class S3CLITarDownloader(IterDataPipe[T_co]):
    def __init__(
        self,
        source_datapipe,
        n_retries: int = 10,
        verbose: bool = False,
        dl_root: str = "/scratch",
        mode: str = "r:*",
        aws_kwargs: Optional[dict] = None,
        ignore_missing_files: bool = False,
    ):
        self.source_datapipe = source_datapipe
        assert os.path.isdir(dl_root), f"`dl_root` {dl_root} is not a valid directory"
        # get slurm job id and create a subdir in `dl_root` to download all the tars to
        dl_root = get_job_temp_dir(dl_root)
        os.makedirs(dl_root, exist_ok=True)
        self.dl_root = dl_root
        self.mode = mode
        self.n_retries = n_retries
        self.ignore_missing_files = ignore_missing_files

        self.verbose = verbose

    def _yield_next(self, url: str, local_path: str):
        if self.verbose:
            _log("popping queue")

        if self.verbose:
            _log(f"loading {local_path}")
        tarstream = tarfile.open(local_path, self.mode)

        if self.verbose:
            _log(f"yielding {url}")

        yield (url, StreamWrapper(tarstream))
        if self.verbose:
            _log("new tar request")

        if self.verbose:
            _log(f"removing previous tar at {local_path}")
        try:
            os.remove(local_path)
        except FileNotFoundError:
            _log(
                f"WARNING: Could not find previous tar for deletion. Unless a clean-up was triggered this is unexpected. The location was {local_path}"
            )

    def __iter__(self):
        with tempfile.TemporaryDirectory(dir=self.dl_root) as scratch:
            for url in self.source_datapipe:
                try:
                    local_path = _download_tar(
                        url, scratch, verbose=self.verbose, n_retries=self.n_retries
                    )
                except FileNotFoundError as e:
                    if not self.ignore_missing_files:
                        raise e
                else:
                    yield from self._yield_next(url, local_path)


@functional_datapipe("load_from_tar_and_handle_error")
class TarArchiveLoaderWithErrorHandlingIterDataPipe(TarArchiveLoader):
    def __init__(
        self,
        datapipe: Iterable[Tuple[str, BufferedIOBase]],
        mode: str = "r:*",
        length: int = -1,
        handler: Callable = wds.reraise_exception,
    ):
        super().__init__(datapipe=datapipe, mode=mode, length=length)
        self.handler = handler

    def __iter__(self) -> Iterator[Tuple[str, BufferedIOBase]]:
        for data in self.datapipe:
            validate_pathname_binary_tuple(data)
            pathname, data_stream = data
            try:
                if isinstance(data_stream, StreamWrapper) and isinstance(
                    data_stream.file_obj, tarfile.TarFile
                ):
                    tar = data_stream.file_obj
                else:
                    reading_mode = (
                        self.mode
                        if hasattr(data_stream, "seekable") and data_stream.seekable()
                        else self.mode.replace(":", "|")
                    )
                    # typing.cast is used here to silence mypy's type checker
                    tar = tarfile.open(
                        fileobj=cast(Optional[IO[bytes]], data_stream),
                        mode=reading_mode,
                    )
                for tarinfo in tar:
                    if not tarinfo.isfile():
                        continue
                    extracted_fobj = tar.extractfile(tarinfo)
                    if extracted_fobj is None:
                        warnings.warn(
                            f"failed to extract file {tarinfo.name} from source tarfile {pathname}"
                        )
                        raise tarfile.ExtractError
                    inner_pathname = os.path.normpath(
                        os.path.join(pathname, tarinfo.name)
                    )

                    yield inner_pathname, StreamWrapper(extracted_fobj, data_stream, name=inner_pathname)  # type: ignore[misc]
            except Exception as e:
                warnings.warn(
                    f"Unable to extract files from corrupted tarfile stream {pathname} due to: {e}, abort!"
                )
                if self.handler(e):
                    if hasattr(e, "args") and len(e.args) > 0:
                        e.args = (e.args[0] + " @ " + str(pathname),) + e.args[1:]
            finally:
                if isinstance(data_stream, StreamWrapper):
                    data_stream.autoclose()

class DictCollator:
    def __init__(
        self,
        combine_tensors: bool = True,
        combine_scalars: bool = True,
        timeout=None,
    ):
        self.combine_tensors = combine_tensors
        self.combine_scalars = combine_scalars

        collate = self._collate
        if timeout is not None:
            collate = timeout_wrapper(collate, timeout=timeout)
        self.collate = collate

    def __call__(self, samples):
        return self.collate(samples)

    def _collate(self, samples):
        keys = set.intersection(*[set(sample.keys()) for sample in samples])
        batched = {key: [] for key in keys}

        for s in samples:
            [batched[key].append(s[key]) for key in batched]

        result = {}
        for key in batched:
            if isinstance(batched[key][0], (int, float)):
                if self.combine_scalars:
                    result[key] = np.array(list(batched[key]))
            elif isinstance(batched[key][0], Tensor):
                if self.combine_tensors:
                    result[key] = stack(list(batched[key]))
            elif isinstance(batched[key][0], np.ndarray):
                if self.combine_tensors:
                    result[key] = np.array(list(batched[key]))
            else:
                result[key] = list(batched[key])

        del samples
        del batched
        return result

def timeout_wrapper(func: Callable, timeout: Optional[float] = None) -> Callable:
    if timeout is None or timeout <= 0.0:
        return func

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        result = [None]
        exception = [None]
        event = threading.Event()

        def wrapped_func():
            try:
                result[0] = func(*args, **kwargs)
            except Exception as e:
                exception[0] = e
            finally:
                event.set()

        thread = threading.Thread(target=wrapped_func)
        thread.start()
        event.wait(timeout)

        if not event.is_set():
            raise TimeoutError(f"Function call timed out (longer than {timeout} secs).")

        thread.join()

        if exception[0] is not None:
            err = exception[0]
            del exception
            raise err

        del thread
        del exception
        del wrapped_func
        del event
        del args
        del kwargs

        ret = result[0]
        del result
        return ret

    return wrapper

class DecoderWithTimeout(wds.Decoder):
    """Decode samples using a list of handlers.

    For each key/data item, this iterates through the list of
    handlers until some handler returns something other than None.
    """

    def __init__(self, *args, timeout: Optional[float] = None, **kwargs):
        super().__init__(*args, **kwargs)

        if timeout is not None:
            self.decode = timeout_wrapper(self.decode, timeout)

def setup_decoder(
    decoders,
    additional_decoder_kwargs: Optional[Union[DictConfig, Dict]] = None,
):
    if not additional_decoder_kwargs:
        additional_decoder_kwargs = {}

    if not isinstance(decoders, (List, ListConfig)):
        decoders = [decoders]
    handlers = []
    for decoder_spec in decoders:
        if isinstance(decoder_spec, (Dict, DictConfig)):
            decoder = instantiate(decoder_spec)
        elif isinstance(decoder_spec, str):
            decoder = wds.autodecode.ImageHandler(decoder_spec)
        else:
            raise TypeError(f"{decoder_spec} not a thing for decoders.")
        handlers.append(decoder)
        print(f"Adding decoder {decoder.__class__.__name__} to decoders.")

    decoder = DecoderWithTimeout(
        handlers,
        partial=additional_decoder_kwargs.pop("partial", True),
        **additional_decoder_kwargs,  # todo
    )
    return decoder

def create_dataset(urls, is_s3=False, prefetch=None, 
                     s3_buffer_size=None, s3_ignore_missing_files=True,
                     sample_shuffle=1, shard_shuffle=1000, cycle=True,
                     split_workers_along_tars=True, decoders="torch"):
    if isinstance(urls, str):
        urls = [urls]
    dp = IterableWrapper(urls)
    dp = dp.custom_shard_expand()
    if is_s3: 
        dp = dp.list_files_with_aws_cli().filter(is_tar)
    else:
        dp = dp.list_files(masks="*.tar", recursive=True)
    
    dp = ShuffledListDataPipe(list(dp), shuffle=shard_shuffle > 1, cycle=cycle)

    if split_workers_along_tars:
        dp = dp.sharding_filter()

    if is_s3:
        # if prefetch is None:
        #     s3_buffer_size = int(s3_buffer_size) if s3_buffer_size is not None else None
        #     dp = dp.wrapped_load_files_by_s3(
        #         buffer_size=s3_buffer_size,
        #         ignore_missing_files=s3_ignore_missing_files,
        #     )
        dp = dp.download_with_s3_cli(ignore_missing_files=s3_ignore_missing_files)
    else:
        dp = dp.open_files(mode="b")

    if prefetch is not None:
        dp = dp.custom_prefetch(buffer_size=prefetch)

    dp = dp.load_from_tar_and_handle_error(handler=error_handler)
    dp = dp.webdataset()

    if not split_workers_along_tars:
        dp = dp.sharding_filter()

    dp = dp.map_with_handler(
            stream_reader,
            handler=error_handler,
            called_cls_name="StreamReader",
        )

    dp = dp.shuffle(buffer_size=sample_shuffle) if sample_shuffle > 1 else dp

    dp = dp.map_with_handler(
            to_our_format,
            handler=error_handler,
            called_cls_name="ToOurFormat",
        )

    decoder: DecoderWithTimeout = setup_decoder(decoders) # ,additional_decoder_kwargs)

    dp = dp.map_with_handler(
        decoder.decode,
        handler=error_handler,
        called_cls_name="Decoder",
    )
    
    return dp

def create_loader(
    datapipeline: IterDataPipe,
    batch_size: int,
    num_workers: int,
    partial: bool = False,
    collation_fn: Optional[Union[Callable, Dict, DictConfig]] = DictCollator(),
    batched_transforms: Optional[ListConfig] = None,
    loader_kwargs: Optional[Union[Dict, DictConfig]] = None,
) -> DataLoader:
    if not loader_kwargs:
        loader_kwargs = {}

    loader_kwargs.pop("shuffle", None)
    if not batched_transforms:
        batched_transforms = []

    print("#" * 100)
    print("Building dataloader with the following parameters")
    print(f"batch_size: {batch_size}, num_workers: {num_workers}")
    for key in loader_kwargs:
        print(key, ": ", loader_kwargs[key])
    print("#" * 100)

    datapipeline = datapipeline.batch(batch_size, drop_last=not partial)

    if isinstance(collation_fn, (Dict, DictConfig)):
        collation_fn = instantiate(collation_fn)
    datapipeline = datapipeline.collate(collate_fn=collation_fn)
    loader = DataLoader(
        datapipeline, batch_size=None, num_workers=num_workers, **loader_kwargs
    )
    return loader

class fMRIDataModule(pl.LightningDataModule):
    def __init__(
        self, 
        train_urls:str,
        test_urls: str,
        batch_size: int,
        num_workers: int
    ):
        super().__init__()
        self.train_urls = train_urls
        self.test_urls = test_urls
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_data = None
        self.test_data = None
    
    def setup(self, stage: str):
        self.train_data = create_dataset(
            self.train_urls,
            is_s3=self.train_urls[:2]=="s3", 
            sample_shuffle=100,
            shard_shuffle=100
         ) if self.train_data is None else self.train_data
        
        self.test_data = create_dataset(
            self.test_urls,
            is_s3=self.test_urls[:2]=="s3", 
            sample_shuffle=1, 
            shard_shuffle=1
        ) if self.test_data is None else self.test_data
        
    def train_dataloader(self):
        return create_loader(self.train_data, batch_size=self.batch_size, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return create_loader(self.test_data, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return self.val_dataloader()