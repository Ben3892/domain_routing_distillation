import numpy as np
import torch
from torch.utils.data import IterableDataset, Sampler, Dataset
from typing import List, Dict, Optional, Iterable
import os
from concurrent.futures import ThreadPoolExecutor
# from megatron.training import get_args

### TODO：后续使用跨分词器蒸馏需要替换
_BEGIN_URL_MARKER_TOKENS = [30, 8277, 24543, 32]# tokenizer.encode("<begin_url>", add_special_tokens=False)
_END_URL_MARKER_TOKENS = [30, 523, 24543, 32]# tokenizer.encode("<end_url>", add_special_tokens=False)

def _mask_url_spans(
    loss_mask: torch.Tensor,
    input_ids: torch.Tensor,
    eos_token_id: int = 1,
    bos_token_id: int = 0,
) -> torch.Tensor:
    """将 input_ids 中所有合法 URL 标记区间（含首尾标记）的 loss_mask 置为 0。

    配对语义：END 配最近的合法 BEGIN；匹配后 END 区间内及之前的 BEGIN 全部废弃。

    Args:
        loss_mask:    shape (seq_len,)，float tensor
        input_ids:    shape (seq_len,)，long tensor
        eos_token_id: EOS token id，可为 None
        bos_token_id: BOS token id，可为 None

    Returns:
        loss_mask（in-place 修改后返回）
    """
    # ---- 异常处理 ----
    if input_ids is None or loss_mask is None:
        return loss_mask
    # ---- 正常逻辑 ----
    seq_len   = input_ids.size(0)
    begin_len = len(_BEGIN_URL_MARKER_TOKENS)
    end_len   = len(_END_URL_MARKER_TOKENS)

    if seq_len < max(begin_len, end_len):
        return loss_mask

    begin_tokens = torch.tensor(
        _BEGIN_URL_MARKER_TOKENS, dtype=input_ids.dtype, device=input_ids.device
    )
    end_tokens = torch.tensor(
        _END_URL_MARKER_TOKENS, dtype=input_ids.dtype, device=input_ids.device
    )

    windows    = input_ids.unfold(0, begin_len, 1)
    begin_hits = (windows == begin_tokens).all(dim=1)
    end_hits   = (windows == end_tokens).all(dim=1)

    all_begin = begin_hits.nonzero(as_tuple=True)[0].tolist()
    end_pos   = end_hits.nonzero(as_tuple=True)[0].tolist()

    # eos/bos 为 None 时自动忽略
    boundary_ids = {tid for tid in (eos_token_id, bos_token_id) if tid is not None}

    begin_pos = [
        bp for bp in all_begin
        if bp == 0 or input_ids[bp - 1].item() in boundary_ids
    ]

    # O(B+E) 双指针
    start_bi = 0
    look_bi  = 0

    for ep in end_pos:
        while look_bi < len(begin_pos) and begin_pos[look_bi] < ep:
            look_bi += 1
        match_i = look_bi - 1
        if match_i >= start_bi:
            bp = begin_pos[match_i]
            loss_mask[bp : ep + end_len] = 0.0
            while look_bi < len(begin_pos) and begin_pos[look_bi] <= ep + end_len - 1:
                look_bi += 1
            start_bi = look_bi

    return loss_mask

class DomainWeightedRandomSampler(Sampler):
    """基于领域权重的随机采样器"""
    
    def __init__(self, total_samples, consumed_samples, micro_batch_size, pre_dataset_len, 
                 data_parallel_rank, data_parallel_size, weights, consumed_per_dataset, ref_loss,
                 domain_names: List[str], drop_last=True):
        """
        Args:
            domain_names: 领域名称列表
            weights: 每个领域的权重
        """
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.consumed_samples = sum(consumed_per_dataset)
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.drop_last = drop_last
        self.ref_loss = ref_loss
        
        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)
        assert self.consumed_samples < self.total_samples, \
            'no samples left to consume: {}, {}'.format(self.consumed_samples,
                                                        self.total_samples)
        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

        assert len(domain_names) == len(weights), "数据集数量必须与数据集权重相同"
        
        self.domain_names = domain_names

        if not weights:
            print("当前混合多个数据集未指定各自权重，请检查是否符合预期")
            self.weights = [1/len(self.domain_names)] * len(self.domain_names)
        else:
            assert len(weights) == len(self.domain_names), "混合多个数据集时，各数据集权重应与数据集数量一致"
            # 权重归一化
            total_weight = sum(weights)
            self.weights = [w / total_weight for w in weights]
        
        assert len(self.ref_loss) == len(self.weights), "混合多个数据集时，各数据集参考损失应与数据集数量一致"

        if consumed_per_dataset:
            self.consumed = consumed_per_dataset
        else:
            self.consumed = [0] * len(self.weights)
        assert len(self.consumed) == len(self.weights), "如果提供了每个数据集的offset，那么长度也应该和权重一样"

        self.pre_dataset_len = pre_dataset_len
        assert len(self.pre_dataset_len) == len(self.weights), "如果提供了每个数据的样本数量，那么样本数量应该和数据集数量相同"

        self.start_idx, self.end_idx = self.get_start_end_idx()

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        # Last batch will be dropped if drop_last is not set False
        offset_in_per_dataset = []
        dataset_ptr = []
        _gen = torch.Generator().manual_seed(42)  # 生成器1

        for _ in range(self.consumed_samples, self.total_samples, self.micro_batch_times_data_parallel_size):
            if sum(self.weights) <= 0:
                print("所有数据集已耗尽，停止采样")
                return
            for _ in range(self.micro_batch_times_data_parallel_size):
                idx = torch.multinomial(
                    torch.tensor(self.weights),
                    num_samples=1,
                    replacement=True,
                    generator=_gen
                ).item()
                dataset_ptr.append(idx)
                offset_in_per_dataset.append(self.consumed[idx])
                self.consumed[idx] += 1
                if self.consumed[idx] >= self.pre_dataset_len[idx]:
                    print(f"数据集{idx}已经完成遍历，后续不再从中采样")
                    self.weights[idx] = 0.0

            return_list = [(a, b) for a, b in zip(dataset_ptr[self.start_idx: self.end_idx], offset_in_per_dataset[self.start_idx: self.end_idx])]
            yield return_list
            dataset_ptr = []
            offset_in_per_dataset = []
            return_list = []
    

    # cyq: 根据验证集的反馈来实现weight的动态更新
    def set_weight(self, loss_for_each_domian: List):
        # delta = [max(i-j, 0)/j for i, j in zip(loss_for_each_domian, self.ref_loss)]
        # alpha = [w * np.exp(d) for w,d in zip(self.weights, delta)]
        # weights = [w / sum(alpha) for w in alpha]
        # for idx, weight in enumerate(weights):
        #     if self.consumed[idx] < self.pre_dataset_len[idx]:
        #         self.weights[idx] = weight
        #     else:
        #         print(f"数据集{idx}已经完成遍历，后续不再从中采样，不对其权重做变更")
        return self.weights

    def __len__(self):
        if self.drop_last:
            return self.total_samples // self.micro_batch_times_data_parallel_size
        else:
            import math
            return math.ceil(self.total_samples / self.micro_batch_times_data_parallel_size)


class NumSequentialSampler(Sampler):
    """基于数量的顺序采样器"""
    
    def __init__(self, total_samples, micro_batch_size, global_batch_size, pre_dataset_len,
                 num_dataset, data_parallel_rank, data_parallel_size):
        """
        Args:
            domain_names: 领域名称列表
            weights: 每个领域的权重
        """
        # Keep a copy of input params for later use.
        self.total_samples = total_samples
        self.micro_batch_size = micro_batch_size
        self.data_parallel_rank = data_parallel_rank
        self.micro_batch_times_data_parallel_size = \
            self.micro_batch_size * data_parallel_size
        self.global_batch_size = global_batch_size
        self.num_dataset = num_dataset
        self.consumed = [0] * num_dataset
        self.pre_dataset_len = pre_dataset_len
        # Sanity checks.
        assert self.total_samples > 0, \
            'no sample to consume: {}'.format(self.total_samples)

        assert self.micro_batch_size > 0
        assert data_parallel_size > 0
        assert self.data_parallel_rank < data_parallel_size, \
            'data_parallel_rank should be smaller than data size: {}, ' \
            '{}'.format(self.data_parallel_rank, data_parallel_size)

        self.start_idx, self.end_idx = self.get_start_end_idx()

    def get_start_end_idx(self):
        start_idx = self.data_parallel_rank * self.micro_batch_size
        end_idx = start_idx + self.micro_batch_size
        return start_idx, end_idx

    def __iter__(self):
        # Last batch will be dropped if drop_last is not set False
        offset_in_per_dataset = []
        dataset_ptr = []
        while True:
            for dataset_index in range(self.num_dataset):
                for _ in range(self.global_batch_size // self.micro_batch_times_data_parallel_size):
                    for _ in range(self.micro_batch_times_data_parallel_size):
                        dataset_ptr.append(dataset_index)
                        self.consumed[dataset_index] = 0 if self.consumed[dataset_index] >= (self.pre_dataset_len[dataset_index] - 1) else self.consumed[dataset_index]
                        offset_in_per_dataset.append(self.consumed[dataset_index])
                        self.consumed[dataset_index] += 1

                    return_list = [(a, b) for a, b in zip(dataset_ptr[self.start_idx: self.end_idx], offset_in_per_dataset[self.start_idx: self.end_idx])]
                    # if torch.distributed.get_rank() == 2:
                    #     print("rank 2", return_list)
                    yield return_list
                    dataset_ptr = []
                    offset_in_per_dataset = []
                    return_list = []

    def __len__(self):
        # 丢弃不完整的 batch
        return self.total_samples // self.global_batch_size


class BlendedDataset(Dataset):
    r"""Dataset as a concatenation of multiple datasets.

    This class is useful to assemble different existing datasets.

    Args:
        datasets (sequence): List of datasets to be concatenated
    """
    def __init__(self, total_samples: int, max_padding_length: int, tokenizer: any, \
    datasets: List[Dataset]=None, datasets_path: List[any]=None, is_mmap: bool=True) -> None:
        
        super().__init__()
        self.total_samples = total_samples
        self.max_lenght = max_padding_length
        self.tokenizer = tokenizer
        assert datasets_path != None or datasets != None, "必须输入多个数据集或者数据集路径进行混合"
        
        if datasets:
            self.datasets = list(datasets)
        elif isinstance(datasets_path[0], dict):
            self.datasets = [NumpyBinDataset(file_path=d_path.get("file_path"), item_length=self.max_lenght, tokenizer=self.tokenizer, is_mmap=is_mmap, quality_id=d_path.get("quality_id", 0)) \
                for d_path in datasets_path]# d_path是一个字段，包含{file_path, quality_id}2个字段
        else: # 之前是List[str]
            self.datasets = [NumpyBinDataset(file_path=d_path, item_length=self.max_lenght, tokenizer=self.tokenizer, is_mmap=is_mmap) \
                for d_path in datasets_path]
        self.pre_dataset_len = [len(dataset) for dataset in self.datasets]
        assert len(self.datasets) > 0, "datasets should not be an empty iterable"  # type: ignore[arg-type]

    def __len__(self):
        return sum(self.pre_dataset_len)

    def get_pre_dataset_len(self):
        return self.pre_dataset_len

    def __getitem__(self, idx):
        dataset_idx, offset_in_per_dataset = idx
        return self.datasets[dataset_idx][offset_in_per_dataset]

class BlendedPTTeacherDataset(Dataset):
    def __init__(self, datasets_path: List[any], max_seq_length: int, file_sizes: Optional[List[int]] = None) -> None:
        """
        Args:
            datasets_path: bin 文件路径列表（不是文件夹）
            file_sizes: 预先获取的文件大小列表，与 datasets_path 一一对应。
                        有则直接用（跳过 stat），无则运行时动态获取。
        """
        super().__init__()
        self.seq_len = max_seq_length

        # 优先使用 config 预存的 file_size，避免启动时重复 AFS stat
        if file_sizes is not None and all(sz is not None for sz in file_sizes):
            resolved_sizes = file_sizes
        else:
            resolved_sizes = self._gather_file_sizes(datasets_path)

        self.datasets = [PTTeacherDataset(file_path=d_path.get("file_path"), seq_length=self.seq_len, is_mmap=False, file_size=sz, quality_id=d_path.get("quality_id", 0)) for d_path, sz in zip(datasets_path, resolved_sizes)]
        self.pre_dataset_len = [len(dataset) for dataset in self.datasets]
        assert len(self.datasets) > 0

    @staticmethod
    def _gather_file_sizes(paths) -> List[int]:
        """rank 0 并行 stat，广播给其他进程。无 distributed 环境时直接并行 stat。"""
        str_paths = [p["file_path"] for p in paths]

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            with ThreadPoolExecutor(max_workers=32) as pool:
                return list(pool.map(os.path.getsize, str_paths))

        rank = torch.distributed.get_rank()
        n = len(str_paths)
        device = torch.cuda.current_device()

        if rank == 0:
            with ThreadPoolExecutor(max_workers=32) as pool:
                sizes = list(pool.map(os.path.getsize, str_paths))
            sizes_tensor = torch.tensor(sizes, dtype=torch.int64, device=device)
        else:
            sizes_tensor = torch.zeros(n, dtype=torch.int64, device=device)

        torch.distributed.broadcast(sizes_tensor, src=0)
        return sizes_tensor.tolist()

    def __len__(self):
        return sum(self.pre_dataset_len)

    def get_pre_dataset_len(self):
        return self.pre_dataset_len

    def __getitem__(self, idx):
        dataset_idx, offset_in_per_dataset = idx
        return self.datasets[dataset_idx][offset_in_per_dataset]

class PTTeacherDataset(Dataset):
    def __init__(self, file_path, seq_length, is_mmap=False, file_size=None, quality_id=0):
        super().__init__()
        self.file_path = file_path
        self.seq_length = seq_length
        self.topk_size = 256
        self.quality_id = quality_id
        
        # 每个样本的字节数
        # 多保存1位token，用来计算lm loss
        self.prompt_bytes  = (seq_length + 1) * 8
        self.logprob_bytes = seq_length * self.topk_size * 4
        self.index_bytes   = seq_length * self.topk_size * 4
        self.item_size     = self.prompt_bytes + self.logprob_bytes + self.index_bytes
        
        # 注意：不用 _FileBinReader（它有重试延迟）
        # 直接用 mmap 或普通文件句柄，快速失败
        if is_mmap:
            self._mmap = np.memmap(self.file_path, mode='r', dtype=np.uint8)
        else:
            self._mmap = None
            # file_size 由调用方提前批量获取并传入，避免每个进程重复 stat
            self._file_size = file_size if file_size is not None else os.path.getsize(self.file_path)
        
        self.dataset_name = file_path.split('env_run/', 1)[1].replace('/', '-').replace('_', '-')

    def __len__(self):
        if self._mmap is not None:
            return self._mmap.size // self.item_size
        return self._file_size // self.item_size

    def __getitem__(self, offset):
        VOCAB_SIZE = 130000
        MAX_RETRIES = 3

        for attempt in range(MAX_RETRIES):
            if len(self)==0:
                continue
            try_offset = (offset + attempt) % len(self)
            try:
                byte_offset = try_offset * self.item_size

                # 用 _mmap 或直接 file read（与 __init__ 保持一致）
                if self._mmap is not None:
                    all_bytes = self._mmap[byte_offset: byte_offset + self.item_size]
                else:
                    with open(self.file_path, 'rb', buffering=0) as f:
                        f.seek(byte_offset)
                        all_bytes = np.frombuffer(f.read(self.item_size), dtype=np.uint8)

                pos = 0
                prompt_size = (self.seq_length + 1) * 8
                prompt = np.frombuffer(all_bytes[pos:pos+prompt_size], dtype=np.int64)
                pos += prompt_size

                if prompt.min() < 0 or prompt.max() >= VOCAB_SIZE:
                    print(f'[WARN] Corrupted token IDs at offset {try_offset} in {self.file_path}, retrying...')
                    continue

                topk_logprob_size = self.seq_length * self.topk_size * 4
                topk_logprob = np.frombuffer(all_bytes[pos:pos+topk_logprob_size], dtype=np.float32)
                topk_logprob = topk_logprob.reshape(self.seq_length, self.topk_size)
                pos += topk_logprob_size

                topk_index_size = self.seq_length * self.topk_size * 4
                topk_index = np.frombuffer(all_bytes[pos:pos+topk_index_size], dtype=np.int32)
                topk_index = topk_index.reshape(self.seq_length, self.topk_size)

                return {
                    'dataset_name': self.dataset_name,
                    'prompt_token_ids': torch.from_numpy(prompt.copy()),
                    'teacher_topk_logprobs': torch.from_numpy(topk_logprob.copy()),
                    'teacher_topk_indices': torch.from_numpy(topk_index.copy()),
                    'quality_id': torch.tensor(self.quality_id, dtype=torch.long),  # scalar → DataLoader 堆叠后 [B]
                }

            except Exception as e:
                print(f'Error reading sample {try_offset} from {self.file_path}: {e}, retrying...')
                continue

        print(f'[ERROR] All {MAX_RETRIES} retries failed for offset {offset} in {self.file_path}')
        return {
            'dataset_name': self.dataset_name,
            'prompt_token_ids': torch.ones(self.seq_length + 1, dtype=torch.long),
            'teacher_topk_logprobs': torch.zeros(self.seq_length, self.topk_size, dtype=torch.float),
            'teacher_topk_indices': torch.zeros(self.seq_length, self.topk_size, dtype=torch.long),
            'quality_id': torch.tensor(self.quality_id, dtype=torch.long),  # 失败时也保持一致
        }

class _BinReader():
    """A _BinReader that memory maps the data (.bin) file

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_buffer_mmap = np.memmap(bin_path, mode="r", order="C")
        self._bin_buffer = memoryview(self._bin_buffer_mmap)

    def get_bin_length(self, dtype, item_length) -> int:
        data_dtype = np.dtype(dtype)
        data_samples = self._bin_buffer_mmap.size // data_dtype.itemsize // item_length
        return data_samples

    def read(self, dtype, count: int, offset: int) -> np.ndarray:
        """Read bytes into a numpy array.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.

            count (int): Number of items to read.

            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items and data-type `dtype` constructed from reading bytes from the data file starting at `offset`.
        """
        return np.frombuffer(self._bin_buffer, dtype=dtype, count=count, offset=offset)

    def __del__(self) -> None:
        """Clean up the object."""
        if self._bin_buffer_mmap is not None:
            self._bin_buffer_mmap._mmap.close()
        del self._bin_buffer_mmap


class _FileBinReader(_BinReader):
    """A _BinReader that reads from the data (.bin) file using a file pointer

    Args:
        bin_path (str): bin_path (str): The path to the data (.bin) file.
    """

    def __init__(self, bin_path: str) -> None:
        self._bin_path = bin_path
        self._file_size = os.path.getsize(bin_path)

    def get_bin_length(self, dtype, item_length) -> int:
        data_dtype = np.dtype(dtype)
        data_samples = self._file_size // data_dtype.itemsize // item_length
        return data_samples

    def read(self, dtype, count: int, offset: int) -> np.ndarray:
        """Read bytes into a numpy array with retry logic.

        Args:
            dtype (Type[numpy.number]): Data-type of the returned array.
            count (int): Number of items to read.
            offset (int): Start reading from this offset (in bytes).

        Returns:
            numpy.ndarray: An array with `count` items.
        """
        max_retries = 5  # 最多重试5次
        retry_delay = 10 # 每次重试前等待10秒
        import time

        for attempt in range(max_retries):
            try:
                sequence = np.empty(count, dtype=dtype)
                # 每次重试都重新打开文件句柄，避免句柄本身的状态问题
                with open(self._bin_path, mode='rb', buffering=0) as bin_buffer_file:
                    bin_buffer_file.seek(offset)
                    bin_buffer_file.readinto(sequence)
                return sequence # 成功读取，直接返回
            except OSError as e:
                # 打印详细的错误日志，方便排查
                print(f"WARNING: Caught OSError in BinReader.read: {e}")
                print(f"File: {self._bin_path}, Offset: {offset}, Count: {count}")
                print(f"Attempt {attempt + 1}/{max_retries}. Retrying in {retry_delay} seconds...")
                
                # 如果是最后一次尝试，就将异常抛出，让上层知道确实失败了
                if attempt + 1 == max_retries:
                    print(f"ERROR: Failed to read data after {max_retries} attempts. Raising exception.")
                    raise e
                
                # 等待一段时间再重试，给网络/文件系统恢复的时间
                time.sleep(retry_delay)
    
    def __del__(self):
        pass

def _get_ltor_masks_ids(
    data: torch.Tensor,
    eod_token: int,
    eod_mask_loss: bool = True,
    reset_position_ids: bool = False,
    reset_attention_mask: bool = True,
    create_attention_mask: bool = False,
):
    """Build masks and position id for left to right model.

    Args:
        data (torch.Tensor): The data tenor that holds the tokens from the dataset
        eod_token (int): ID of the token to that is considered the EOD
        eod_mask_loss (bool): Switch to enable the EOD mask loss
        create_attention_mask (bool): Switch to enable the attention masks generation. Can be
            disabled if attention kernel generates masks by itself.

    Returns:
        torch.Tensor: Attention mask needed to be used for Attention
        torch.Tensor: The mask used for loss value during training
        torch.Tensor: The position ID's of the token
    """
    seq_length = data.numel()

    if create_attention_mask:
        attention_mask = torch.tril(
            torch.ones((seq_length, seq_length), device=data.device)
        ).unsqueeze(0)  # (1, seq_length, seq_length)
    else:
        attention_mask = None

    # Loss mask.
    loss_mask = torch.ones(seq_length, dtype=torch.float, device=data.device)
    if eod_mask_loss:
        loss_mask[data == eod_token] = 0.0

    # Position ids.
    position_ids = torch.arange(seq_length, dtype=torch.long, device=data.device)
    if reset_position_ids:
        position_ids = position_ids.clone()

    # cyq：这儿即使重设了attention_mask，因为flash_attn不接受这样的形式也不会生效
    if reset_position_ids or reset_attention_mask:
        # Find indices where EOD token is.
        eod_index = position_ids[data == eod_token]
        # Detach indices from positions if going to modify positions.
        if reset_position_ids:
            eod_index = eod_index.clone()

        # Loop through EOD indices:
        prev_index = 0
        for j in range(eod_index.numel()):
            i = eod_index[j]
            # Mask attention loss.
            if reset_attention_mask and attention_mask is not None:
                attention_mask[0, (i + 1) :, : (i + 1)] = 0
            # Reset positions.
            if reset_position_ids:
                position_ids[(i + 1) :] -= i + 1 - prev_index
                prev_index = i + 1

    if attention_mask is not None:
        # Convert attention mask to binary:
        attention_mask = attention_mask < 0.5

    return attention_mask, loss_mask, position_ids


class NumpyBinDataset(Dataset):
    def __init__(self, file_path, item_length, tokenizer, dtype=np.int32, is_mmap=True, quality_id=0):
        """
        基于Dataset的与Numpy映射读取二进制bin文件的数据集
        
        参数:
            file_path (str): numpy二进制文件路径
            item_length (int): 每个样本的长度(数据点数)
            dtype: numpy数据类型(默认为np.int32)
            pad_token_id: label末尾的pad_token_id，100001对应deepseek-v2-lite的
        """
        self.file_path = file_path
        self.dataset_name = file_path.split('env_run/', 1)[1].replace('/', '-').replace('_', '-')
        self.item_length = item_length
        self.dtype = dtype
        data_dtype = np.dtype(dtype)
        self.item_size = data_dtype.itemsize

        if is_mmap:
            self.bin_reader = _BinReader(self.file_path)
        else:
            self.bin_reader = _FileBinReader(self.file_path)
        self.tokenizer = tokenizer
        self._pad_token_id = self.tokenizer.pad_token_id

    def __len__(self):
        return self.bin_reader.get_bin_length(dtype=self.dtype, item_length=self.item_length)

    def __getitem__(self, offset):
        # 计算偏移量，随后取出来
        # 注意：这里需要读取item_length + 1个token，因为要获取最后一个位置的label，
        # 但是offset只偏移item_length个长度
        sequence = self.bin_reader.read(
                dtype=self.dtype, count=(self.item_length + 1), offset=offset * self.item_length * self.item_size
            )
        # 最后一个位置label可以是pad_token，但是大概率不是
        text = torch.from_numpy(sequence.copy()).long()
        # labels = torch.roll(input_ids, shifts=-1, dims=0)
        # labels[-1] = self._pad_token_id
        input_ids = text[:-1].contiguous()
        labels = text[1:].contiguous()
        # bug1: 最后一个位置label不应该设置为pad token
        # labels[-1] = self._pad_token_id
        attention_mask, loss_mask, position_ids = _get_ltor_masks_ids(
            input_ids,
            self.tokenizer.eod,
            eod_mask_loss=True,
            reset_attention_mask=True,
            create_attention_mask=False,
            reset_position_ids=False
        )
        #bug2: 对于input_ids==pad_token的mask掉，而不是对label进行mask,
        # _get_ltor_masks_ids 函数已经处理，这里不再做处理
        # loss_mask[input_ids == self._pad_token_id] = 0.0
        # input_ids, labels=bos的部分mask掉
        # bug3: <begin_url>...<end_url>之间的内容没有mask掉
        if input_ids is not None:
            loss_mask[input_ids == 0] = 0.0
            loss_mask[labels == 0] = 0.0
            loss_mask = _mask_url_spans(loss_mask, input_ids)
        # For padded sequences, ensure the embedding layer can map the token ID
        # bug4: 不用把bos替换为eos，避免teacher ood 
        # input_ids[input_ids == self._pad_token_id] = 0
        # labels[labels == self._pad_token_id] = 0
        
        if offset is None:
            loss_mask = torch.zeros_like(loss_mask)
        quality_id_tensor = torch.tensor([self.quality_id], dtype=torch.long)
        # 将quality_id转化为tensor形式
        train_sample = {
            'tokens': input_ids,
            'labels': labels,
            'loss_mask': loss_mask,
            'position_ids': position_ids,
            'dataset_name': self.dataset_name,
            'quality_id': quality_id_tensor
        }
        return train_sample

if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("deepseek/deepseek_v3")
    batch_sampler = DomainWeightedRandomSampler(
            total_samples=5000,
            consumed_samples=0,
            micro_batch_size=2,
            data_parallel_rank=1,
            data_parallel_size=2,
            weights=[3, 3, 3],
            consumed_per_dataset=[0, 0, 0],
            domain_names=["1", "2", "3"]
        )
    datasets_path = []
    dataset = BlendedDataset(datasets_path=datasets_path, total_samples=100, max_padding_length=100, tokenizer=tokenizer)
    train_dataloader = torch.utils.data.DataLoader(dataset,
                                       batch_sampler=batch_sampler,
                                       num_workers=20,
                                       pin_memory=True,
                                       persistent_workers=True,
                                       )
    for i, input_ids in enumerate(train_dataloader):
        print(f"{i}th: input_ids: ", input_ids)
        tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist()[0])
        print("Tokens:", tokens)  # 例如 ['[CLS]', 'hello', 'world', '!', '[SEP]']
        # 如果想合并成完整字符串（注意subword标记）
        text_reconstructed = tokenizer.convert_tokens_to_string(tokens)
        print("Reconstructed text:", text_reconstructed)  # 例如 "[CLS] hello world! [SEP]"
        if i >= 1:
            break
        