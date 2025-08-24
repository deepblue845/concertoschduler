import torch
import time
import heapq
import threading
import queue
from collections import deque
import numpy as np
import csv
import random
from datetime import datetime
import matplotlib.pyplot as plt

class MetricsCollector:
    def __init__(self):
        self.data = {
            'timestamp': [],
            'kv_cache_usage': [],
            'online_queue': [],
            'offline_queue': [],
            'active_requests': [],
            'throughput_tokens': [],
            'throughput_requests': [],
            'preemption_count': [],
            'evicted_tokens': [],
            'cycle_time': [],
            'ttft': [],
            'tbt': [],
            'scheduled_online': [],
            'scheduled_offline': [],
            'worker_queue_sizes': {0: [], 1: []},
            'checkpoint_operations': [],
            'safepoint_checks': []
        }
        self.start_time = time.time()
        self.last_tokens = 0
        self.last_requests = 0
        self.last_timestamp = self.start_time
    
    def record(self, scheduler, cycle_time, scheduled_online=0, scheduled_offline=0):
        timestamp = time.time() - self.start_time
        self.data['timestamp'].append(timestamp)
        
        # KV Cache metrics with boundary protection
        current_usage = scheduler.kv_cache.current_usage()
        max_size = scheduler.kv_cache.max_size
        kv_usage = min(1.0, current_usage / max_size)  # Ensure never exceeds 1.0
        
        # Log warnings for high memory usage
        if current_usage > max_size:
            print(f"⚠️ KV Cache overflow detected! Usage: {current_usage}/{max_size}")
        elif current_usage > 0.9 * max_size:
            print(f"⚠️ KV Cache near full! Usage: {current_usage}/{max_size} ({kv_usage*100:.1f}%)")
            
        self.data['kv_cache_usage'].append(kv_usage)
        
        # Queue metrics
        self.data['online_queue'].append(len(scheduler.online_queue))
        self.data['offline_queue'].append(len(scheduler.offline_queue))
        self.data['active_requests'].append(len(scheduler.active_requests))
        
        # Preemption metrics
        self.data['preemption_count'].append(scheduler.stats['preemption_count'])
        self.data['evicted_tokens'].append(scheduler.stats['evicted_tokens'])
        
        # Throughput metrics
        delta_time = timestamp - self.last_timestamp
        if delta_time > 0:
            token_throughput = (scheduler.stats['generated_tokens'] - self.last_tokens) / delta_time
            request_throughput = (scheduler.stats['completed_requests'] - self.last_requests) / delta_time
        else:
            token_throughput = 0
            request_throughput = 0
            
        self.data['throughput_tokens'].append(token_throughput)
        self.data['throughput_requests'].append(request_throughput)
        self.data['cycle_time'].append(cycle_time)
        
        # Scheduling metrics
        self.data['scheduled_online'].append(scheduled_online)
        self.data['scheduled_offline'].append(scheduled_offline)
        
        # Worker queue metrics
        for i, worker in enumerate(scheduler.workers):
            if i not in self.data['worker_queue_sizes']:
                self.data['worker_queue_sizes'][i] = []
            self.data['worker_queue_sizes'][i].append(worker.batch_queue.qsize())
        
        # Update last values
        self.last_tokens = scheduler.stats['generated_tokens']
        self.last_requests = scheduler.stats['completed_requests']
        self.last_timestamp = timestamp
    
    def record_request_metrics(self, request):
        if request.first_token_time and request.start_time:
            ttft = (request.first_token_time - request.start_time) * 1000
            self.data['ttft'].append(ttft)
        
        if request.last_token_time and request.first_token_time and len(request.output_tokens) > 1:
            tbt = (request.last_token_time - request.first_token_time) * 1000 / max(1, len(request.output_tokens) - 1)
            self.data['tbt'].append(tbt)
    
    def record_checkpoint(self, tokens):
        self.data['checkpoint_operations'].append(tokens)
    
    def record_safepoint(self):
        self.data['safepoint_checks'].append(time.time() - self.start_time)
    
    def plot_metrics(self):
        print("\nGenerating performance metrics visualizations...")
        plt.figure(figsize=(25, 35))
        
        # KV Cache Usage
        plt.subplot(6, 2, 1)
        plt.plot(self.data['timestamp'], self.data['kv_cache_usage'])
        plt.title('KV Cache Usage')
        plt.xlabel('Time (s)')
        plt.ylabel('Usage Ratio')
        plt.axhline(y=0.7, color='r', linestyle='--', label='Pressure Threshold (70%)')
        plt.grid(True)
        
        """# Queue Lengths
        plt.subplot(6, 2, 2)
        plt.plot(self.data['timestamp'], self.data['online_queue'], label='Online Queue')
        plt.plot(self.data['timestamp'], self.data['offline_queue'], label='Offline Queue')
        plt.plot(self.data['timestamp'], self.data['active_requests'], label='Active Requests')
        plt.title('Queue Lengths')
        plt.xlabel('Time (s)')
        plt.ylabel('Count')
        plt.legend()
        plt.grid(True)"""
        
        
        # Scheduling Metrics
        plt.subplot(6, 2, 2)
        plt.plot(self.data['timestamp'], self.data['scheduled_online'], label='Online Tokens')
        plt.plot(self.data['timestamp'], self.data['scheduled_offline'], label='Offline Tokens')
        plt.title('Scheduled Tokens per Cycle')
        plt.xlabel('Time (s)')
        plt.ylabel('Tokens')
        plt.legend()
        plt.grid(True)
        
        # Cycle Time
        plt.subplot(6, 2, 3)
        plt.plot(self.data['timestamp'], self.data['cycle_time'])
        plt.title('Scheduling Cycle Time')
        plt.xlabel('Time (s)')
        plt.ylabel('Cycle Time (ms)')
        plt.grid(True)
        
        
        """# TTFT Distribution
        if self.data['ttft']:
            plt.subplot(6, 2, 8)
            plt.hist(self.data['ttft'], bins=20, alpha=0.7)
            plt.title('TTFT Distribution')
            plt.xlabel('TTFT (ms)')
            plt.ylabel('Count')
            plt.grid(True)
        
        # TBT Distribution
        if self.data['tbt']:
            plt.subplot(6, 2, 9)
            plt.hist(self.data['tbt'], bins=20, alpha=0.7)
            plt.title('TBT Distribution')
            plt.xlabel('TBT (ms/token)')
            plt.ylabel('Count')
            plt.grid(True)
        
        # TTFT Over Time
        if self.data['ttft']:
            plt.subplot(6, 2, 10)
            plt.plot(range(len(self.data['ttft'])), self.data['ttft'], 'o-')
            plt.title('TTFT Over Time')
            plt.xlabel('Request Index')
            plt.ylabel('TTFT (ms)')
            plt.grid(True)
        
        # TBT Over Time
        if self.data['tbt']:
            plt.subplot(6, 2, 11)
            plt.plot(range(len(self.data['tbt'])), self.data['tbt'], 'o-')
            plt.title('TBT Over Time')
            plt.xlabel('Request Index')
            plt.ylabel('TBT (ms/token)')
            plt.grid(True)
        
        # Checkpoint Operations
        if self.data['checkpoint_operations']:
            plt.subplot(6, 2, 12)
            plt.plot(range(len(self.data['checkpoint_operations'])), self.data['checkpoint_operations'], 'o-')
            plt.title('Checkpoint Operations')
            plt.xlabel('Operation Index')
            plt.ylabel('Tokens Checkpointed')
            plt.grid(True)
        """
        
        
        # 调整布局参数 - 解决重叠问题
        plt.tight_layout(pad=4.0, w_pad=2.0, h_pad=3.0)
        plt.subplots_adjust(top=0.95)  # 为标题留出空间
        
        plt.savefig('concerto_metrics.png')
        print("Metrics visualization saved to concerto_metrics.png")
        plt.show()

class KVCache:
    def __init__(self, max_size, key_dim, value_dim):
        self.max_size = max_size
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.keys = torch.empty((max_size, key_dim), dtype=torch.float32, device='cuda')
        self.values = torch.empty((max_size, value_dim), dtype=torch.float32, device='cuda')
        self.current_size = 0
        self.token_map = {}  # request_id -> (start_idx, end_idx)
        self.host_checkpoints = {}  # For incremental checkpointing
        self.checkpoint_stream = torch.cuda.Stream()  # Separate stream for async checkpointing
        self.lock = threading.Lock()  # Add lock for thread safety
        self.pending_checkpoints = {}  # Track pending checkpoints
        self.checkpoint_rate = 0  # Adaptive checkpointing rate
        self.checkpoint_threshold = 0.7  # Start checkpointing when usage > 70%
        self.last_checkpoint_time = time.time()

    def add_sequence(self, request_id, keys, values):
        with self.lock:
            seq_len = keys.shape[0]
            # Ensure we have enough space
            if self.current_size + seq_len > self.max_size:
                raise RuntimeError(f"KV Cache overflow: {self.current_size}+{seq_len} > {self.max_size}")
                
            start_idx = self.current_size
            end_idx = start_idx + seq_len
            
            self.keys[start_idx:end_idx] = keys
            self.values[start_idx:end_idx] = values
            self.token_map[request_id] = (start_idx, end_idx)
            self.current_size = end_idx
            
            # Immediately checkpoint if needed
            self._adaptive_checkpoint(request_id, keys, values)
            
            return start_idx, end_idx

    def get_sequence(self, request_id):
        with self.lock:
            if request_id not in self.token_map:
                return None, None
                
            start, end = self.token_map[request_id]
            return self.keys[start:end], self.values[start:end]

    def evict_request(self, request_id, save_to_host=True):
        with self.lock:
            if request_id not in self.token_map:
                return 0
                
            start, end = self.token_map[request_id]
            seq_len = end - start
            
            # Save to host checkpoint before eviction
            if save_to_host:
                if request_id not in self.host_checkpoints:
                    self.host_checkpoints[request_id] = {
                        'keys': self.keys[start:end].cpu().clone(),
                        'values': self.values[start:end].cpu().clone(),
                        'length': seq_len
                    }
                else:
                    # Update only the new part (incremental checkpointing)
                    existing_len = self.host_checkpoints[request_id]['length']
                    if existing_len < seq_len:
                        new_keys = self.keys[start+existing_len:end].cpu().clone()
                        new_values = self.values[start+existing_len:end].cpu().clone()
                        self.host_checkpoints[request_id]['keys'] = torch.cat([
                            self.host_checkpoints[request_id]['keys'], new_keys
                        ])
                        self.host_checkpoints[request_id]['values'] = torch.cat([
                            self.host_checkpoints[request_id]['values'], new_values
                        ])
                        self.host_checkpoints[request_id]['length'] = seq_len
            
            # Shift subsequent blocks to fill the gap
            if end < self.current_size:
                self.keys[start:start + (self.current_size - end)] = self.keys[end:self.current_size]
                self.values[start:start + (self.current_size - end)] = self.values[end:self.current_size]
                
            # Update token mappings
            del self.token_map[request_id]
            for rid, (s, e) in list(self.token_map.items()):
                if s >= end:  # Only update blocks starting after evicted block
                    self.token_map[rid] = (s - seq_len, e - seq_len)
            
            self.current_size -= seq_len
            return seq_len

    def restore_request(self, request_id):
        """Restore a request from host checkpoint"""
        with self.lock:
            if request_id not in self.host_checkpoints:
                return False
                
            checkpoint = self.host_checkpoints[request_id]
            keys = checkpoint['keys'].to('cuda')
            values = checkpoint['values'].to('cuda')
            seq_len = keys.shape[0]
            
            if self.current_size + seq_len > self.max_size:
                return False
                
            start_idx = self.current_size
            end_idx = start_idx + seq_len
            self.keys[start_idx:end_idx] = keys
            self.values[start_idx:end_idx] = values
            self.token_map[request_id] = (start_idx, end_idx)
            self.current_size = end_idx
            
            return True

    def current_usage(self):
        with self.lock:
            return self.current_size

    def available_space(self):
        with self.lock:
            return self.max_size - self.current_size

    def checkpoint_new_tokens(self, request_id, new_keys, new_values):
        """Incremental checkpointing of new tokens"""
        with self.lock:
            if request_id not in self.host_checkpoints:
                self.host_checkpoints[request_id] = {
                    'keys': new_keys.cpu().clone(),
                    'values': new_values.cpu().clone(),
                    'length': new_keys.shape[0]
                }
            else:
                # Asynchronous checkpointing in background stream
                with torch.cuda.stream(self.checkpoint_stream):
                    self.host_checkpoints[request_id]['keys'] = torch.cat([
                        self.host_checkpoints[request_id]['keys'], new_keys.cpu().clone()
                    ])
                    self.host_checkpoints[request_id]['values'] = torch.cat([
                        self.host_checkpoints[request_id]['values'], new_values.cpu().clone()
                    ])
                    self.host_checkpoints[request_id]['length'] += new_keys.shape[0]
    
    def _adaptive_checkpoint(self, request_id, keys, values):
        """Adaptive checkpointing based on memory pressure"""
        usage_ratio = self.current_size / self.max_size
        if usage_ratio > self.checkpoint_threshold:
            # Increase checkpoint rate as pressure increases
            pressure = min(1.0, (usage_ratio - self.checkpoint_threshold) / (1.0 - self.checkpoint_threshold))
            self.checkpoint_rate = pressure
            
            # Checkpoint new tokens
            self.checkpoint_new_tokens(request_id, keys, values)
            
            # Log checkpoint operation
            return keys.shape[0]
        return 0

    def get_host_checkpoint(self, request_id):
        with self.lock:
            if request_id in self.host_checkpoints:
                return self.host_checkpoints[request_id]
            return None

    def clear_host_checkpoint(self, request_id):
        with self.lock:
            if request_id in self.host_checkpoints:
                del self.host_checkpoints[request_id]

class Request:
    def __init__(self, request_id, prompt, max_output_len, is_online=False, slo_ttft=None, slo_tbt=None):
        self.id = request_id
        self.prompt = prompt
        self.prompt_len = len(prompt)
        self.max_output_len = max_output_len
        self.is_online = is_online
        self.slo_ttft = slo_ttft
        self.slo_tbt = slo_tbt
        
        # State tracking
        self.generated_tokens = 0
        self.output_tokens = []  # Store generated tokens
        self.prefill_done = False
        self.start_time = time.time()
        self.first_token_time = None
        self.last_token_time = None
        self.kv_cache_position = None
        self.last_preemption_time = 0
        self.preemption_count = 0
        self.current_layer = 0  # Track current layer
        self.is_scheduled_in_current_cycle = False
        self.lock = threading.Lock()  # Add lock for thread safety
        self.last_safepoint = 0  # Last layer where safepoint was checked

    def get_next_tokens(self, max_tokens):
        with self.lock:
            if self.prefill_done:
                # Decode phase: process one token at a time
                if len(self.output_tokens) < self.max_output_len:
                    # Return the last generated token
                    input_token = self.output_tokens[-1] if self.output_tokens else self.prompt[-1]
                    return [input_token], True  # True indicates decode phase
                return None, True
            else:
                # Prefill phase: process chunks of tokens
                start = self.generated_tokens
                end = min(start + max_tokens, self.prompt_len)
                tokens = self.prompt[start:end]
                self.generated_tokens = end  # Update the generated tokens count
                
                # Check if prefill is completed
                if self.generated_tokens >= self.prompt_len:
                    self.prefill_done = True
                    
                return tokens, False  # False indicates prefill phase
    
    def add_output_token(self, token):
        """Add newly generated output token"""
        with self.lock:
            self.output_tokens.append(token)
            if not self.first_token_time:
                self.first_token_time = time.time()
            self.last_token_time = time.time()
            self.generated_tokens = self.prompt_len + len(self.output_tokens)
    
    def current_context_size(self):
        return self.prompt_len + len(self.output_tokens)
    
    def is_completed(self):
        return self.prefill_done and len(self.output_tokens) >= self.max_output_len
    
    def __repr__(self):
        status = "online" if self.is_online else "offline"
        phase = "decode" if self.prefill_done else "prefill"
        progress = f"{len(self.output_tokens)}/{self.max_output_len}" if self.prefill_done else f"{self.generated_tokens}/{self.prompt_len}"
        return f"Request(id={self.id}, status={status}, phase={phase}, progress={progress}, ctx={self.current_context_size()})"

class Worker(threading.Thread):
    def __init__(self, kv_cache, worker_id, num_layers=32, layers_per_safepoint=8):
        super().__init__(daemon=True)
        self.kv_cache = kv_cache
        self.worker_id = worker_id
        self.num_layers = num_layers
        self.layers_per_safepoint = layers_per_safepoint
        self.current_batch = None
        self.current_layer = 0
        self.preemption_signal = False
        self.batch_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.active_requests = {}  # Track active requests being processed
        self.preempted_requests = []  # Track preempted requests for later resumption
        self.last_safepoint_check = 0
        self.safepoint_overhead = 0.00035  # 350 microseconds per safepoint

    def run(self):
        while not self.stop_event.is_set():
            try:
                batch = self.batch_queue.get(timeout=0.1)
                if batch is None:  # Sentinel value
                    break
                    
                self.current_batch = batch
                self.execute_batch(batch)
                self.current_batch = None
            except queue.Empty:
                continue
    
    def execute_batch(self, batch):
        """Execute batch with layer-wise preemption"""
        print(f"\n[Worker {self.worker_id}] Starting batch execution at {datetime.now().strftime('%H:%M:%S.%f')}")
        print(f"[Worker {self.worker_id}] Batch contains {len(batch)} requests:")
        for i, (req, tokens, is_decode) in enumerate(batch):
            phase = "decode" if is_decode else "prefill"
            print(f"  {i+1}. {req} - Processing {len(tokens)} tokens ({phase})")
        
        with self.lock:
            self.preemption_signal = False
            self.current_layer = 0
            
            # Add sequences to KV cache (prefill phase)
            for req, tokens, is_decode in batch:
                if not is_decode:  # Prefill phase
                    try:
                        # Convert tokens to tensor (simulated)
                        keys = torch.randn(len(tokens), self.kv_cache.key_dim, device='cuda')
                        values = torch.randn(len(tokens), self.kv_cache.value_dim, device='cuda')
                        
                        # Ensure we have enough space
                        if len(tokens) > self.kv_cache.available_space():
                            print(f"[Worker {self.worker_id}] Insufficient space for request {req.id} ({len(tokens)} tokens)")
                            continue
                            
                        start, end = self.kv_cache.add_sequence(req.id, keys, values)
                        req.kv_cache_position = (start, end)
                        req.generated_tokens = end  # Update generated tokens
                        print(f"[Worker {self.worker_id}] Added {len(tokens)} tokens to KV cache for request {req.id} (position: {start}-{end})")
                    except RuntimeError as e:
                        print(f"[Worker {self.worker_id}] Failed to add sequence to KV cache: {e}")
                        continue
        
        # Track active requests in this batch
        with self.lock:
            self.active_requests = {req.id: req for req, _, _ in batch}
        
        # Execute layers with safepoints for preemption
        for layer in range(self.num_layers):
            # Check for preemption at safepoints
            if layer % self.layers_per_safepoint == 0:
                time.sleep(self.safepoint_overhead)  # Simulate safepoint overhead
                self.metrics.record_safepoint()
                
                if self.check_preemption():
                    print(f"[Worker {self.worker_id}] Preemption signal received at safepoint (layer {layer})")
                    self.handle_preemption(batch)
                    return False
            
            # Execute layer
            print(f"[Worker {self.worker_id}] Executing layer {layer+1}/{self.num_layers}")
            time.sleep(0.01)  # 10ms per layer
            self.current_layer = layer
            
            # Check if any requests are completed
            with self.lock:
                completed_reqs = []
                for req_id, req in self.active_requests.items():
                    if req.is_completed():
                        completed_reqs.append(req_id)
                
                # Remove completed requests
                for req_id in completed_reqs:
                    del self.active_requests[req_id]
        
        # For decode phase, add the generated token to the request
        with self.lock:
            for req, tokens, is_decode in batch:
                if is_decode and tokens:
                    # Only generate output if we have tokens
                    req.add_output_token(tokens[0])
                    print(f"[Worker {self.worker_id}] Generated token for request {req.id} - {req}")
                    
                    # For offline requests, checkpoint new token
                    if not req.is_online:
                        try:
                            start, end = req.kv_cache_position
                            # Extend KV cache for new token
                            new_key = torch.randn(1, self.kv_cache.key_dim, device='cuda')
                            new_value = torch.randn(1, self.kv_cache.value_dim, device='cuda')
                            
                            # Check if we have space for new token
                            if self.kv_cache.available_space() < 1:
                                print(f"[Worker {self.worker_id}] Insufficient space for new token in request {req.id}")
                            else:
                                # Add new token to KV cache
                                self.kv_cache.keys[end] = new_key
                                self.kv_cache.values[end] = new_value
                                req.kv_cache_position = (start, end + 1)
                                self.kv_cache.current_size += 1
                                
                                # Checkpoint new token
                                self.kv_cache.checkpoint_new_tokens(req.id, new_key, new_value)
                                print(f"[Worker {self.worker_id}] Checkpointed new token for offline request {req.id}")
                        except Exception as e:
                            print(f"[Worker {self.worker_id}] Error during decode: {e}")
        
        print(f"[Worker {self.worker_id}] Batch execution completed at {datetime.now().strftime('%H:%M:%S.%f')}")
        return True
    
    def check_preemption(self):
        """Check if preemption is requested at safepoint"""
        with self.lock:
            return self.preemption_signal
    
    def handle_preemption(self, batch):
        """Handle preemption by cleaning up offline requests"""
        print(f"[Worker {self.worker_id}] Handling preemption for batch")
        # Clean up and evict offline requests
        for req, _, _ in batch:
            if not req.is_online:
                try:
                    tokens_freed = self.kv_cache.evict_request(req.id)
                    if req.id in self.active_requests:
                        del self.active_requests[req.id]
                    req.kv_cache_position = None
                    req.preemption_count += 1
                    req.last_preemption_time = time.time()
                    self.preempted_requests.append(req)
                    print(f"[Worker {self.worker_id}] Preempted offline request {req.id} (freed {tokens_freed} tokens)")
                except Exception as e:
                    print(f"[Worker {self.worker_id}] Error evicting request {req.id}: {e}")
    
    def signal_preemption(self):
        """Signal worker to preempt at next safepoint"""
        with self.lock:
            self.preemption_signal = True
    
    def stop_worker(self):
        self.stop_event.set()
        self.batch_queue.put(None)  # Sentinel value

class Profiler:
    def __init__(self):
        # Simulated profile data: (token_count, context_length) -> latency_ms
        self.profile_data = {
            (64, 256): 15,
            (128, 512): 25,
            (256, 1024): 45,
            (512, 2048): 78,
            (1024, 4096): 145,
            (2048, 8192): 280,
        }
    
    def tbt_to_token_budget(self, tbt_slo_ms):
        """Convert TBT SLO to token budget"""
        # Find maximum token count that meets the TBT SLO
        max_tokens = 0
        for (tokens, ctx_len), latency in self.profile_data.items():
            if latency <= tbt_slo_ms and tokens > max_tokens:
                max_tokens = tokens
        return max(64, min(2048, max_tokens))  # Keep within reasonable bounds
    
    def estimate_execution_time(self, token_count, context_length):
        """Estimate execution time for a batch"""
        # Find closest profile point
        closest_key = min(self.profile_data.keys(), 
                         key=lambda x: abs(x[0]-token_count) + abs(x[1]-context_length))
        base_time = self.profile_data.get(closest_key, 100)  # Default 100ms
        
        # Convert to seconds
        estimated_time = base_time / 1000.0
        return estimated_time

class ConcertoScheduler:
    def __init__(self, kv_cache, profiler, tbt_slo_ms, num_workers=1):
        self.kv_cache = kv_cache
        self.profiler = profiler
        self.tbt_slo_ms = tbt_slo_ms
        
        # Request queues
        self.online_queue = deque()    # High priority
        self.offline_queue = deque()   # Low priority
        self.active_requests = {}      # ID -> Request (both online and offline)
        self.preempted_offline_requests = []  # Track preempted offline requests
        
        # Preemption state
        self.preemption_signal = False
        self.ttft_preemption_signal = False
        self.running_batch = None
        
        # Configuration
        self.mem_threshold = int(0.7 * kv_cache.max_size)  # 70% threshold as in paper
        self.max_chunk_size = 2048
        self.preemption_overhead = 0.02  # 20ms preemption overhead
        self.ttft_slo_scale = 1.25  # Allowed latency increase over baseline
        self.metrics = MetricsCollector()
        # Create workers
        self.workers = []
        for i in range(num_workers):
            worker = Worker(kv_cache, worker_id=i)
            worker.metrics = self.metrics  # Share metrics collector
            worker.start()
            self.workers.append(worker)
        
        # Background thread for TTFT monitoring
        self.monitor_thread = threading.Thread(target=self._ttft_monitor_loop, daemon=True)
        self.monitor_active = True
        self.monitor_event = threading.Event()
        self.monitor_thread.start()
        
        # Background thread for memory pressure handling
        self.memory_pressure_thread = threading.Thread(target=self._handle_memory_pressure_loop, daemon=True)
        self.memory_pressure_active = True
        self.memory_pressure_event = threading.Event()
        self.memory_pressure_thread.start()
        
        # Background thread for checkpoint prefetching
        self.prefetch_thread = threading.Thread(target=self._prefetch_loop, daemon=True)
        self.prefetch_active = True
        self.prefetch_event = threading.Event()
        self.prefetch_thread.start()
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'completed_requests': 0,
            'generated_tokens': 0,
            'evicted_tokens': 0,
            'preemption_count': 0,
            'scheduling_cycles': 0,
            'ttft_violations': 0,
            'tbt_violations': 0,
            'safepoint_checks': 0
        }
        
        # Metrics collection
        
    
    def stop(self):
        """Stop the scheduler and its background threads"""
        self.monitor_active = False
        self.monitor_event.set()
        self.monitor_thread.join(timeout=1.0)
        
        self.memory_pressure_active = False
        self.memory_pressure_event.set()
        self.memory_pressure_thread.join(timeout=1.0)
        
        self.prefetch_active = False
        self.prefetch_event.set()
        self.prefetch_thread.join(timeout=1.0)
        
        for worker in self.workers:
            worker.stop_worker()
            worker.join()
        
        print("\n=== Simulation Statistics ===")
        print(f"Total requests processed: {self.stats['total_requests']}")
        print(f"Completed requests: {self.stats['completed_requests']}")
        print(f"Generated tokens: {self.stats['generated_tokens']}")
        print(f"Evicted tokens: {self.stats['evicted_tokens']}")
        print(f"Preemption events: {self.stats['preemption_count']}")
        print(f"Scheduling cycles: {self.stats['scheduling_cycles']}")
        print(f"TTFT violations: {self.stats['ttft_violations']}")
        print(f"TBT violations: {self.stats['tbt_violations']}")
        print(f"Current KV cache usage: {self.kv_cache.current_usage()}/{self.kv_cache.max_size} tokens")
        print(f"Active requests: {len(self.active_requests)}")
        
        # Generate metrics visualization
        self.metrics.plot_metrics()
    
    def _ttft_monitor_loop(self):
        """Background thread for TTFT violation detection"""
        while self.monitor_active:
            # Check every 50ms
            self.monitor_event.wait(0.05)
            if not self.monitor_active:
                break
            self.monitor_event.clear()
            
            if not self.online_queue or not self.running_batch:
                continue
            
            self._detect_ttft_violation()
    
    def _handle_memory_pressure_loop(self):
        """Background thread for handling memory pressure"""
        while self.memory_pressure_active:
            self.memory_pressure_event.wait(0.2)  # Check every 200ms
            if not self.memory_pressure_active:
                break
            self.memory_pressure_event.clear()
            
            self._handle_memory_pressure()
    
    def _prefetch_loop(self):
        """Background thread for prefetching preempted requests"""
        while self.prefetch_active:
            self.prefetch_event.wait(0.5)  # Check every 500ms
            if not self.prefetch_active:
                break
            self.prefetch_event.clear()
            
            self._prefetch_preempted_requests()
    
    def _detect_ttft_violation(self):
        """Detect potential TTFT violation"""
        if not self.running_batch or not self.online_queue:
            return
            
        # Estimate remaining time for current batch
        current_batch_time = self.profiler.estimate_execution_time(
            self.running_batch['token_count'], 
            self.running_batch['context_length']
        )
        
        # Estimate time for queued online requests
        queued_time = 0
        for req in list(self.online_queue) + list(self.active_requests.values()):
            if req.is_online and not req.prefill_done:
                # Estimate prefill time for new requests
                prefill_time = self.profiler.estimate_execution_time(
                    req.prompt_len, 
                    req.prompt_len
                )
                queued_time += prefill_time
                
        total_estimated_time = current_batch_time + queued_time
        
        # Get most stringent TTFT SLO from online requests
        min_ttft_slo = min(
            (req.slo_ttft for req in self.online_queue if req.slo_ttft is not None),
            default=1000  # Default 1000ms if no SLO specified
        )
        
        # Apply scaling to SLO
        scaled_ttft_slo = min_ttft_slo * self.ttft_slo_scale
        
        if total_estimated_time > scaled_ttft_slo / 1000.0:  # Convert ms to seconds
            # Signal for TTFT preemption
            print(f"[TTFT Monitor] Potential TTFT violation detected! Estimated time: {total_estimated_time:.3f}s, SLO: {scaled_ttft_slo/1000:.3f}s")
            self.ttft_preemption_signal = True
            self.stats['ttft_violations'] += 1
            self.signal_preemption()
    
    def _handle_memory_pressure(self):
        """Evict offline requests when memory pressure is high"""
        current_usage = self.kv_cache.current_usage()
        if current_usage < self.mem_threshold:
            return 0
        
        print(f"[Memory] Pressure detected! Usage: {current_usage}/{self.kv_cache.max_size} (threshold: {self.mem_threshold})")
        
        # Sort offline requests by context length (shortest first)
        evict_candidates = []
        for req in self.active_requests.values():
            if not req.is_online and req.kv_cache_position is not None:
                # Prefer requests that are short and have been preempted less
                score = req.current_context_size() / (req.preemption_count + 1)
                heapq.heappush(evict_candidates, (score, req.id, req))
        
        freed = 0
        while evict_candidates and self.kv_cache.current_usage() > self.mem_threshold:
            _, _, req = heapq.heappop(evict_candidates)
            try:
                tokens_freed = self.kv_cache.evict_request(req.id)
                freed += tokens_freed
                self.stats['evicted_tokens'] += tokens_freed
                req.kv_cache_position = None
                # Move to preempted queue if not completed
                if not req.is_completed():
                    self.preempted_offline_requests.append(req)
                print(f"[Memory] Evicted request {req.id} (freed {tokens_freed} tokens)")
            except Exception as e:
                print(f"[Memory] Error evicting request {req.id}: {e}")
        
        return freed
    
    def _prefetch_preempted_requests(self):
        """Prefetch preempted requests when resources are available"""
        if not self.preempted_offline_requests or self.kv_cache.current_usage() > 0.5 * self.kv_cache.max_size:
            return
        
        print(f"[Prefetch] Prefetching preempted requests (current usage: {self.kv_cache.current_usage()}/{self.kv_cache.max_size})")
        
        # Try to restore preempted requests
        restored = []
        for req in list(self.preempted_offline_requests):
            try:
                if self.kv_cache.restore_request(req.id):
                    restored.append(req)
                    print(f"[Prefetch] Restored request {req.id} from checkpoint")
            except Exception as e:
                print(f"[Prefetch] Error restoring request {req.id}: {e}")
        
        # Remove restored requests from preempted list
        for req in restored:
            self.preempted_offline_requests.remove(req)
            self.offline_queue.appendleft(req)  # Add back to offline queue
    
    def add_request(self, request):
        """Add a new request to the system"""
        self.stats['total_requests'] += 1
        if request.is_online:
            self.online_queue.append(request)
            # Trigger immediate TTFT check
            self.monitor_event.set()
        else:
            self.offline_queue.append(request)
        self.active_requests[request.id] = request
        print(f"[Scheduler] Added request: {request}")
    
    def _schedule_online(self, token_budget):
        """Schedule online requests within token budget"""
        batch = []
        scheduled_tokens = 0

        print("[Scheduler] Scheduling online requests...")
        for req in list(self.active_requests.values()):
            if not req.is_online or req.is_completed() or req.is_scheduled_in_current_cycle:
                continue

            max_tokens = min(token_budget - scheduled_tokens, self.max_chunk_size)
            tokens, is_decode = req.get_next_tokens(max_tokens)

            if tokens:
                # Estimate memory needed: prefill needs full tokens, decode needs only 1
                mem_needed = len(tokens) if not is_decode else 1
                available = self.kv_cache.available_space()

                # Ensure we have space for this request
                if mem_needed <= available:
                    batch.append((req, tokens, is_decode))
                    scheduled_tokens += len(tokens)
                    req.is_scheduled_in_current_cycle = True
                    phase = "decode" if is_decode else "prefill"
                    print(f"  Scheduled request {req.id} ({phase}, {len(tokens)} tokens, mem_needed: {mem_needed}, available: {available})")
                    if scheduled_tokens >= token_budget:
                        break
                else:
                    print(f"  Skipped request {req.id} (insufficient memory: needed {mem_needed}, available {available})")

        print(f"[Scheduler] Scheduled {len(batch)} online requests ({scheduled_tokens} tokens)")
        return batch, scheduled_tokens
    
    def _schedule_offline(self, remaining_budget):
        """Schedule offline requests with remaining token budget"""
        batch = []
        scheduled_tokens = 0

        print("[Scheduler] Scheduling offline requests...")
        # Schedule active offline requests
        for req in list(self.active_requests.values()):
            if req.is_online or req.is_completed() or req.is_scheduled_in_current_cycle:
                continue

            max_tokens = min(remaining_budget - scheduled_tokens, self.max_chunk_size)
            tokens, is_decode = req.get_next_tokens(max_tokens)

            if tokens:
                # Estimate memory needed
                mem_needed = len(tokens) if not is_decode else 1
                available = self.kv_cache.available_space()

                if mem_needed <= available:
                    batch.append((req, tokens, is_decode))
                    scheduled_tokens += len(tokens)
                    req.is_scheduled_in_current_cycle = True
                    phase = "decode" if is_decode else "prefill"
                    print(f"  Scheduled request {req.id} ({phase}, {len(tokens)} tokens, mem_needed: {mem_needed}, available: {available})")
                    if scheduled_tokens >= remaining_budget:
                        break
                else:
                    print(f"  Skipped request {req.id} (insufficient memory: needed {mem_needed}, available {available})")

        # Schedule new offline requests
        while self.offline_queue and scheduled_tokens < remaining_budget:
            req = self.offline_queue.popleft()
            if req.is_scheduled_in_current_cycle:
                continue

            max_tokens = min(remaining_budget - scheduled_tokens, self.max_chunk_size)
            tokens, is_decode = req.get_next_tokens(max_tokens)

            if tokens:
                # Estimate memory needed
                mem_needed = len(tokens) if not is_decode else 1
                available = self.kv_cache.available_space()

                if mem_needed <= available:
                    batch.append((req, tokens, is_decode))
                    scheduled_tokens += len(tokens)
                    req.is_scheduled_in_current_cycle = True
                    phase = "decode" if is_decode else "prefill"
                    print(f"  Scheduled NEW request {req.id} ({phase}, {len(tokens)} tokens, mem_needed: {mem_needed}, available: {available})")
                    if scheduled_tokens >= remaining_budget:
                        break
                else:
                    print(f"  Skipped NEW request {req.id} (insufficient memory: needed {mem_needed}, available {available})")

        print(f"[Scheduler] Scheduled {len(batch)} offline requests ({scheduled_tokens} tokens)")
        return batch, scheduled_tokens
    
    def _handle_preemption(self):
        """Preempt offline requests and manage KV caches"""
        self.stats['preemption_count'] += 1
        print(f"[Preemption] Handling preemption event (#{self.stats['preemption_count']})")
        # Signal workers to preempt at next safepoint
        for worker in self.workers:
            worker.signal_preemption()
        
        # Wait for preemption to complete
        time.sleep(self.preemption_overhead)
    
    def schedule(self):
        """Main scheduling algorithm"""
        self.stats['scheduling_cycles'] += 1
        cycle_start = time.time()
        print(f"\n=== Scheduling Cycle #{self.stats['scheduling_cycles']} at {datetime.now().strftime('%H:%M:%S.%f')} ===")
        print(f"[State] Online queue: {len(self.online_queue)}, Offline queue: {len(self.offline_queue)}, Active requests: {len(self.active_requests)}")
        print(f"[KV Cache] Usage: {self.kv_cache.current_usage()}/{self.kv_cache.max_size} tokens ({self.kv_cache.available_space()} available)")
        
        # Reset scheduling flags for all requests
        for req in self.active_requests.values():
            req.is_scheduled_in_current_cycle = False
            
        # Step 1: Handle memory pressure
        space_freed = self._handle_memory_pressure()
        if space_freed > 0:
            print(f"[Memory] Evicted {space_freed} tokens from offline requests")
        
        # Step 2: Convert TBT SLO to token budget
        token_budget = self.profiler.tbt_to_token_budget(self.tbt_slo_ms)
        print(f"[Budget] TBT SLO: {self.tbt_slo_ms}ms -> Token budget: {token_budget}")
        
        # Step 3: Schedule online requests first
        online_batch, scheduled_online = self._schedule_online(token_budget)
        remaining_budget = token_budget - scheduled_online
        print(f"[Scheduler] Online scheduled: {scheduled_online} tokens, Remaining budget: {remaining_budget}")
        
        # Step 4: Schedule offline requests if resources available
        offline_batch, scheduled_offline = [], 0

        if remaining_budget > 0 and not self.preemption_signal:
            offline_batch, scheduled_offline = self._schedule_offline(remaining_budget)
        
        # Step 5: Special handling when no online requests
        if not online_batch and not offline_batch:
            print("[Scheduler] No requests scheduled, switching to offline max throughput mode")
            # Max throughput mode for offline requests
            offline_batch, scheduled_offline = self._schedule_offline(self.max_chunk_size)
        
        # Combine batches
        full_batch = online_batch + offline_batch
        total_tokens = scheduled_online + scheduled_offline
        
        # Record running batch for TTFT monitoring
        if full_batch:
            self.running_batch = {
                'start_time': time.time(),
                'token_count': total_tokens,
                'context_length': max(r.current_context_size() for r, _, _ in full_batch),
                'is_online': any(r.is_online for r, _, _ in full_batch)
            }
            print(f"[Scheduler] Prepared batch with {len(full_batch)} requests ({total_tokens} tokens)")
        else:
            self.running_batch = None
            print("[Scheduler] No requests to schedule")
        
        # Handle preemption signals
        if self.preemption_signal:
            self._handle_preemption()
            self.preemption_signal = False
        
        # Handle TTFT preemption signal
        if self.ttft_preemption_signal:
            self.ttft_preemption_signal = False
            print("[TTFT] Handling TTFT preemption signal")
            # For TTFT preemption, prioritize online requests
            if offline_batch:
                # Requeue offline requests
                for req, _, _ in offline_batch:
                    if not req.is_completed() and not req.is_online:
                        self.offline_queue.appendleft(req)
                # Return only online batch
                full_batch = online_batch
                print(f"[TTFT] Requeued {len(offline_batch)} offline requests")
        
        # Clean up completed requests
        completed_ids = [req_id for req_id, req in list(self.active_requests.items()) if req.is_completed()]
        for req_id in completed_ids:
            self.stats['completed_requests'] += 1
            total_tokens = req.prompt_len + len(req.output_tokens)
            self.stats['generated_tokens'] += total_tokens
            print(f"[Cleanup] Request {req_id} completed (tokens: {total_tokens})")
            
            # Record request metrics
            self.metrics.record_request_metrics(req)
            
            if req_id in self.kv_cache.token_map:
                try:
                    self.kv_cache.evict_request(req_id, save_to_host=False)  # No need to save completed requests
                except Exception as e:
                    print(f"[Cleanup] Error evicting completed request {req_id}: {e}")
            del self.active_requests[req_id]
        
        # Prefetch preempted requests if resources are available
        self.prefetch_event.set()
        
        # Dispatch batch to worker
        if full_batch:
            # Find least busy worker
            worker = min(self.workers, key=lambda w: w.batch_queue.qsize())
            worker.batch_queue.put(full_batch)
            print(f"[Dispatch] Sent batch to worker {worker.worker_id} (queue size: {worker.batch_queue.qsize()})")
        
        cycle_time = (time.time() - cycle_start) * 1000
        
        # Record metrics for this cycle
        self.metrics.record(self, cycle_time, scheduled_online, scheduled_offline)
        
        return full_batch
    
    def signal_preemption(self):
        """Trigger preemption for offline requests"""
        print("[Preemption] Preemption signal triggered")
        self.preemption_signal = True


if __name__ == "__main__":
    print("=== Concerto Simulation Started ===")
    print("Initializing KV cache and scheduler...")

    # Initialize components with larger cache
    kv_cache = KVCache(max_size=500000, key_dim=128, value_dim=128)  # 500k tokens cache
    profiler = Profiler()
    scheduler = ConcertoScheduler(kv_cache, profiler, tbt_slo_ms=100, num_workers=2)
    data1 = np.random.gamma(shape=2, scale=50, size=20)
    arr=[int(x)for x in data1]
    arr1=[]
    arr1.append(arr[0])
    for i in range(9):
        arr1.append(arr1[i]+arr[i+1])
    print(arr1)
    data2=np.random.normal(loc=8000,scale=1400,size=10)
    arr2=[int(x)for x in data2]
    print(arr2)
    burstgpt=open('BurstGPT_1.csv','r',newline='',encoding='utf-8')
    reader=csv.reader(burstgpt)
    requests=[]
    for timetag in range(len(arr1)):
        requests.append((arr1[timetag]/10.0,arr2[timetag],int(0.6*arr2[timetag]),False))
    i=0
    for row in reader:
        if i>0:
            requests.append((int(row[0])/10.0,int(row[2]),int(row[3]),True))
        i+=1
        if i>1000:
            break
    requests.sort(key=lambda x:x[0])
    # Simulation loop
    request_index = 0
    start_time = time.time()
    simulation_duration = 200  # seconds

    print("Starting simulation...")
    while time.time() - start_time < simulation_duration:
        current_time = time.time() - start_time

        # Add new requests that arrive at this time
        while request_index < len(requests) and requests[request_index][0] <= current_time:
            _, prompt_len, output_len, is_online = requests[request_index]
            req = Request(
                request_id=request_index + 1,
                prompt=list(range(prompt_len)),
                max_output_len=output_len,
                is_online=is_online,
                slo_ttft=2000,
                slo_tbt=100
            )
            scheduler.add_request(req)
            request_index += 1

        # Schedule requests
        batch = scheduler.schedule()
        time.sleep(0.1)  # Scheduling cycle

    # Clean up
    print("Simulation completed, stopping scheduler...")
    scheduler.stop()
    print("=== Concerto Simulation Ended ===")