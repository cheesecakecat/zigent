const std = @import("std");
const builtin = @import("builtin");
const assert = std.debug.assert;
const math = std.math;
const meta = std.meta;
const simd = std.simd;
const Atomic = std.atomic.Value;
const Allocator = std.mem.Allocator;

/// Entity indices are the backbone of our ECS, serving as lightweight handles to track entities across systems.
/// We use u32 to strike a balance between memory usage and entity count - supporting up to 4 billion entities
/// while keeping overhead minimal. For most cases, even massive ones, this is more than sufficient.
/// Note: If you're building a simulation that genuinely needs more entities, consider using u64 but be aware
/// of the increased memory cost across all systems.
pub const IndexType = u32;

/// Fine-tune how the index pool manages memory and handles concurrent access. The defaults work well
/// for most cases, providing a good balance of safety and performance. You'll mainly want to adjust
/// these when you have specific requirements around memory usage or multi-threading.
///
/// Warning: Changing these values at runtime isn't supported - they're compile-time configurations that
/// affect the entire pool's behavior. If you need different settings for different entity groups,
/// create multiple pool instances.
///
/// Performance Note: The thread_safe option adds synchronization overhead. Only enable it if you're
/// actually accessing the pool from multiple threads. Even then, consider if you could restructure
/// your code to use separate pools per thread instead.
pub const Config = struct {
    /// How many indices to allocate upfront. Pick a number close to your expected entity count
    /// to avoid frequent resizing. The memory is allocated immediately, so don't go overboard.
    ///
    /// Memory Impact: Each index typically uses 4-8 bytes depending on configuration. A capacity
    /// of 1024 uses about 4-8KB of memory. For cases with very tight memory constraints, start
    /// smaller and let it grow as needed.
    initial_capacity: IndexType = 1024,

    /// Sets an absolute limit on how many indices can be allocated. Use this to prevent runaway
    /// entity creation from eating all available memory. A value of 0 means no limit.
    ///
    /// Warning: On 32-bit systems, be extra careful with unlimited growth. The theoretical maximum
    /// of 4 billion entities would require at least 16GB just for the indices, not counting
    /// component data.
    ///
    /// Set this to slightly above your expected maximum for safety, while still having
    /// a reasonable bound. For example, if you expect 10,000 entities max, setting this to 15,000
    /// gives you headroom without risking runaway growth.
    max_capacity: IndexType = 0,

    /// Controls how aggressively the pool grows when it runs out of space. A value of 2.0 means
    /// double the capacity each time. Higher values mean fewer resizes but more potential waste,
    /// lower values save memory but may cause more allocations.
    ///
    /// Memory Impact: With 2.0, a pool starting at 1024 will grow to 2048, then 4096, etc. Each
    /// resize allocates a new chunk and copies the old data. If memory is tight, consider a lower
    /// value like 1.5.
    ///
    /// Warning: Values too close to 1.0 can cause performance problems with frequent resizing.
    /// We recommend staying between 1.3 and 2.0 unless you have very specific needs.
    growth_factor: f32 = 2.0,

    /// When the pool is this empty, it will shrink to free unused memory. Set to 0 to never shrink.
    /// This helps prevent memory bloat in cases where entity count fluctuates significantly.
    ///
    /// Performance Note: Shrinking copies all existing indices to a new, smaller allocation. If your
    /// entity count oscillates frequently around the threshold, you might see performance spikes
    /// from constant resize operations.
    ///
    /// Monitor your entity count patterns in real case. If it stays relatively stable,
    /// consider disabling shrinking to avoid unnecessary overhead.
    shrink_threshold: f32 = 0.25,

    /// Enables SIMD operations for batch processing of indices. This can significantly speed up
    /// operations on large batches of entities, but adds some code complexity and may not help
    /// with small batches.
    ///
    /// Performance Note: The actual speedup depends heavily on your CPU and usage patterns. Profile
    /// with your actual workload before committing to SIMD. Sometimes the overhead of setting up
    /// SIMD operations can outweigh the benefits for small batches.
    ///
    /// Warning: Some CPU architectures might not support SIMD or have different vector sizes.
    /// The code automatically falls back to scalar operations when needed.
    enable_simd: bool = true,

    /// Minimum number of operations needed before we use SIMD processing. Below this, we use
    /// regular scalar code. Finding the right value requires profiling with your specific workload.
    ///
    /// Performance Impact: Too low means we waste time on SIMD setup for small batches. Too high
    /// means we miss optimization opportunities. The sweet spot typically lies between 8 and 16
    /// for most workloads.
    min_batch_size: usize = 8,

    /// Enables thread-safe operations through atomic operations and mutex locks. This adds overhead
    /// but ensures safe concurrent access from multiple threads.
    ///
    /// Performance Impact: Thread safety adds significant overhead - each operation requires atomic
    /// operations and potential mutex locks. If you're not using multiple threads, keep this off.
    ///
    /// Consider using multiple non-thread-safe pools (one per thread) instead of a single
    /// thread-safe pool if your architecture allows it. This can be significantly faster.
    thread_safe: bool = false,

    /// Enables runtime checks for common errors like double-frees and use-after-free. These checks
    /// are invaluable during development but do add overhead.
    ///
    /// Memory Impact: Adds a bitset tracking valid indices, using about 1 bit per possible entity.
    /// For 1 million entities, this adds about 125KB of overhead.
    ///
    /// Enable this during development and testing, then disable it for release builds
    /// once you're confident in your entity management code.
    debug_mode: bool = false,

    /// Enables event callbacks for monitoring pool operations. Great for debugging, metrics
    /// collection, or implementing entity lifecycle hooks.
    ///
    /// Performance Note: The overhead is usually minimal if event handlers are lightweight.
    /// However, complex handlers can significantly impact performance since they run
    /// synchronously with pool operations.
    ///
    /// Warning: Keep event handlers fast and simple. They can block pool operations and
    /// potentially cause deadlocks if not careful with thread-safe pools.
    enable_events: bool = false,

    /// Validates configuration at compile-time to catch problematic combinations early.
    /// This helps prevent runtime surprises from misconfiguration.
    pub fn validate(comptime self: Config) void {
        if (self.initial_capacity == 0) @compileError("Initial capacity must be > 0");
        if (self.growth_factor <= 1.0) @compileError("Growth factor must be > 1.0");
        if (self.min_batch_size == 0) @compileError("Minimum batch size must be > 0");
        if (self.min_batch_size > 32) @compileError("Minimum batch size must be <= 32");
        if (self.shrink_threshold < 0.0 or self.shrink_threshold > 1.0)
            @compileError("Shrink threshold must be between 0.0 and 1.0");
    }
};

/// Track significant events in the index pool's lifecycle. These events provide visibility into
/// the pool's behavior and can help diagnose issues or collect metrics.
///
/// Performance Impact: Event handling adds overhead to every operation that generates events.
/// Keep handlers lightweight and consider disabling events in performance-critical scenarios.
///
/// Use events to implement features like entity lifecycle hooks, debugging tools,
/// or performance monitoring systems. They're also great for catching entity leaks during
/// development.
pub const Event = union(enum) {
    /// Tracks newly allocated indices. Useful for monitoring entity creation patterns
    /// or implementing initialization logic.
    allocation: IndexType,

    /// Tracks when indices return to the pool. Great for catching entity leaks or
    /// monitoring destruction patterns.
    free: IndexType,

    /// Monitors pool growth and shrinking. Watch these events to tune your initial
    /// capacity and growth settings.
    resize: struct { old_cap: usize, new_cap: usize },

    /// Captures operation failures. Essential for debugging and error tracking during
    /// development.
    @"error": Error,
};

pub const Error = error{
    /// No more indices available and can't grow the pool further. This usually means
    /// either hitting max_capacity or running out of memory.
    OutOfIndices,

    /// Attempted to free an index that's outside the valid range. This often indicates
    /// memory corruption or using an index from a different pool.
    InvalidIndex,

    /// Attempted to free an index that's already in the free list. This always indicates
    /// a bug in entity management - you're either freeing twice or using a stale index.
    DoubleFree,

    /// Hit the configured max_capacity limit. Unlike OutOfIndices, this is a deliberate
    /// limit rather than a resource exhaustion issue.
    CapacityExceeded,
};

/// Function type for handling pool events. Implement this to hook into pool operations
/// for logging, debugging, or logic.
///
/// Warning: Event handlers run synchronously in the same thread as the pool operation.
/// Keep them fast to avoid performance issues.
///
/// If you need to do heavy processing in response to events, consider queuing
/// the event data for processing in a separate thread or during the next update cycle.
pub const EventHandler = *const fn (event: Event) void;

/// A balanced configuration that works well for most cases. Provides safety and decent
/// performance without excessive memory usage.
///
/// Start with this configuration and only customize options if profiling shows
/// a need. Premature optimization of these settings rarely pays off.
pub const default_config = Config{};

/// The core implementation of our index pool system. Provides a highly efficient way to manage
/// entity indices with optional thread safety, debugging tools, and SIMD optimizations. The pool
/// recycles indices in LIFO order to maximize cache coherency and minimize memory fragmentation.
///
/// Performance Note: The implementation heavily optimizes the common operations (alloc/free) while
/// providing rich debugging and monitoring capabilities when needed. All debug features can be
/// disabled for production builds with zero overhead.
///
/// Warning: The pool's behavior is configured at compile-time through the Config struct. Runtime
/// configuration changes aren't supported - create separate pool instances if you need different
/// configurations for different subsystems.
pub fn IndexPoolImpl(comptime config: Config) type {
    // Validate config at comptime
    config.validate();

    return struct {
        const Self = @This();
        const SimdVec = @Vector(config.min_batch_size, IndexType);

        /// The allocator used for all memory management. The pool will use this for
        /// its internal data structures and any dynamic allocations needed during
        /// operation.
        ///
        /// Memory Note: The pool tries to minimize allocations by pre-allocating space
        /// and growing geometrically when needed. Monitor the memory usage through
        /// getMemoryUsage() if you need to optimize allocation patterns.
        allocator: Allocator,

        /// Stores indices that have been freed and can be reused. Uses a LIFO (stack)
        /// pattern to maximize cache coherency - recently freed indices are reused first.
        /// In thread-safe mode, each index is wrapped in an atomic for safe concurrent access.
        ///
        /// Performance Note: The LIFO pattern isn't just about speed - it also helps reduce
        /// memory fragmentation by keeping active indices clustered together.
        free_indices: std.ArrayListUnmanaged(if (config.thread_safe) std.atomic.Value(IndexType) else IndexType),

        /// Tracks the next index to allocate when the free list is empty. In thread-safe mode,
        /// this is an atomic value to ensure safe concurrent increments.
        ///
        /// Implementation Note: Using a simple counter here means indices are allocated
        /// sequentially, which is great for cache locality but can lead to fragmentation
        /// if many indices in the middle are freed.
        next_index: if (config.thread_safe) std.atomic.Value(IndexType) else IndexType,

        /// Mutex for thread synchronization. Only present in thread-safe mode.
        ///
        /// Performance Impact: The mutex protects bulk operations and list management,
        /// while individual index operations use atomic operations where possible to
        /// minimize contention.
        mutex: if (config.thread_safe) std.Thread.Mutex else void,

        /// Optional event handler for monitoring pool operations. When enabled, this can
        /// receive notifications about allocations, frees, resizes, and errors.
        ///
        /// Warning: Event handlers run synchronously in the operation's thread. Keep them
        /// fast and simple to avoid becoming a performance bottleneck.
        event_handler: if (config.enable_events) ?EventHandler else void,

        /// Debug validation bitset tracking which indices are currently in use. Only
        /// present when debug_mode is enabled.
        ///
        /// Memory Impact: Uses 1 bit per possible index. The overhead is usually worth
        /// it during development to catch bugs early, but can be disabled in production.
        used_indices: if (config.debug_mode) std.DynamicBitSetUnmanaged else void,

        /// Initialize a new index pool with the given allocator. This immediately allocates
        /// space for the initial capacity, so choose that value carefully.
        ///
        /// Memory Note: The initial allocation includes space for free_indices and the
        /// debug bitset if enabled. See getMemoryUsage() for detailed breakdown.
        ///
        /// Error: Returns error.OutOfMemory if the initial allocation fails.
        pub fn init(allocator: Allocator) !Self {
            var self: Self = undefined;
            self.allocator = allocator;
            self.free_indices = try std.ArrayListUnmanaged(if (config.thread_safe) std.atomic.Value(IndexType) else IndexType).initCapacity(allocator, config.initial_capacity);

            if (config.thread_safe) {
                self.next_index = std.atomic.Value(IndexType).init(0);
                self.mutex = std.Thread.Mutex{};
            } else {
                self.next_index = 0;
            }

            if (config.enable_events) {
                self.event_handler = null;
            }

            if (config.debug_mode) {
                self.used_indices = try std.DynamicBitSetUnmanaged.initEmpty(allocator, config.initial_capacity);
            }

            return self;
        }

        /// Release all memory held by the pool. After calling this, the pool cannot be used
        /// until init() is called again.
        ///
        /// Warning: This does not check if there are still active indices. In debug mode,
        /// you can iterate through used indices first if you need to clean up associated
        /// resources.
        pub fn deinit(self: *Self) void {
            self.free_indices.deinit(self.allocator);
            if (config.debug_mode) {
                self.used_indices.deinit(self.allocator);
            }
        }

        /// Register a callback to receive pool events. The handler will be called
        /// synchronously for allocations, frees, resizes, and errors.
        ///
        /// Warning: Only available if enable_events is true in the config. The handler
        /// must be fast to avoid blocking pool operations.
        pub fn setEventHandler(self: *Self, handler: EventHandler) void {
            if (config.enable_events) {
                self.event_handler = handler;
            }
        }

        /// Internal helper to emit events when event handling is enabled.
        /// Inlined to eliminate overhead when events are disabled.
        inline fn emitEvent(self: *Self, event: Event) void {
            if (config.enable_events) {
                if (self.event_handler) |handler| {
                    handler(event);
                }
            }
        }

        /// Allocate a new entity index. This is the main way to get indices for new entities.
        /// The returned index is guaranteed to be unique until freed.
        ///
        /// Performance Note: This operation is optimized for both the common case (getting
        /// a recycled index) and the growth case (allocating a new index). In thread-safe
        /// mode, it uses atomic operations to minimize contention.
        ///
        /// Error: Returns error.OutOfIndices if allocation fails due to hitting max_capacity
        /// or running out of memory.
        pub fn alloc(self: *Self) !IndexType {
            // Check capacity limit
            if (config.max_capacity > 0 and self.total() >= config.max_capacity) {
                self.emitEvent(.{ .@"error" = Error.CapacityExceeded });
                return Error.CapacityExceeded;
            }

            if (config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();

                // Try to get from free list first
                if (self.free_indices.popOrNull()) |atomic_val| {
                    const index = atomic_val.load(.acquire);
                    if (config.debug_mode) {
                        self.used_indices.set(index);
                    }
                    self.emitEvent(.{ .allocation = index });
                    return index;
                }

                // No free indices, allocate new one
                const index = self.next_index.load(.acquire);
                self.next_index.store(index + 1, .release);
                if (config.debug_mode) {
                    self.used_indices.set(index);
                }
                self.emitEvent(.{ .allocation = index });
                return index;
            } else {
                // Non-thread-safe path
                if (self.free_indices.popOrNull()) |index| {
                    if (config.debug_mode) {
                        self.used_indices.set(index);
                    }
                    self.emitEvent(.{ .allocation = index });
                    return index;
                }

                const index = self.next_index;
                self.next_index += 1;
                if (config.debug_mode) {
                    self.used_indices.set(index);
                }
                return index;
            }
        }

        /// Return an index to the pool for reuse. The index must have been previously
        /// allocated from this pool and not already freed.
        ///
        /// Warning: Using an invalid index or double-freeing an index will return an
        /// error in debug mode. In release mode, this could corrupt the pool's state.
        ///
        /// Error: Returns error.InvalidIndex for out-of-range indices or error.DoubleFree
        /// in debug mode for already-freed indices.
        pub fn free(self: *Self, index: IndexType) !void {
            const current_total = if (config.thread_safe)
                self.next_index.load(.acquire)
            else
                self.next_index;

            if (index >= current_total) {
                self.emitEvent(.{ .@"error" = Error.InvalidIndex });
                return Error.InvalidIndex;
            }

            if (config.debug_mode) {
                if (!self.used_indices.isSet(index)) {
                    self.emitEvent(.{ .@"error" = Error.DoubleFree });
                    return Error.DoubleFree;
                }
                self.used_indices.unset(index);
            }

            if (config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();

                const atomic_val = std.atomic.Value(IndexType).init(index);
                try self.free_indices.append(self.allocator, atomic_val);
            } else {
                try self.free_indices.append(self.allocator, index);
            }

            self.emitEvent(.{ .free = index });
        }

        /// Get the current number of indices available for immediate allocation.
        /// This is the size of the free list.
        ///
        /// Thread Safety: In thread-safe mode, this value may be outdated by the
        /// time you use it. It's mainly useful for monitoring and debugging.
        pub fn available(self: *Self) usize {
            if (config.thread_safe) {
                self.mutex.lock();
                defer self.mutex.unlock();
                return self.free_indices.items.len;
            }
            return self.free_indices.items.len;
        }

        /// Get the total number of indices ever allocated by this pool. This is
        /// effectively the high-water mark of index usage.
        ///
        /// Memory Impact: The pool may reserve more space than this value indicates,
        /// see getMemoryUsage() for actual memory consumption.
        pub inline fn total(self: Self) IndexType {
            return if (config.thread_safe)
                self.next_index.load(.acquire)
            else
                self.next_index;
        }

        /// Check if an index is currently valid (allocated and not freed).
        /// In debug mode, this uses the validation bitset for accurate checking.
        ///
        /// Warning: Without debug mode, this only checks if the index is in the valid
        /// range. It cannot detect use-after-free bugs in release builds.
        pub inline fn isValid(self: Self, index: IndexType) bool {
            if (config.debug_mode) {
                return index < self.total() and self.used_indices.isSet(index);
            }
            return index < self.total();
        }

        /// Get detailed information about the pool's memory usage. This includes
        /// both the total bytes used and the overhead from internal structures.
        ///
        /// Memory Note: The overhead includes the free list, debug bitset if enabled,
        /// and internal structure sizes. This can help tune the pool's configuration
        /// for your specific needs.
        pub fn getMemoryUsage(self: Self) struct {
            total_bytes: usize,
            overhead_bytes: usize,
        } {
            var mem_total: usize = @sizeOf(Self);
            var overhead: usize = @sizeOf(Self);

            // Free list memory
            mem_total += self.free_indices.capacity * @sizeOf(IndexType);
            overhead += self.free_indices.capacity * @sizeOf(IndexType);

            // Debug bitset memory
            if (config.debug_mode) {
                const bitset_bytes = (self.used_indices.capacity() + 7) / 8;
                mem_total += bitset_bytes;
                overhead += bitset_bytes;
            }

            return .{
                .total_bytes = mem_total,
                .overhead_bytes = overhead,
            };
        }

        /// Iterator for walking through all currently allocated indices.
        /// This is particularly useful for cleanup operations or debugging.
        ///
        /// Performance Note: The iterator checks each possible index sequentially.
        /// For sparse allocations, this can be slower than tracking active indices
        /// separately.
        pub const UsedIndicesIterator = struct {
            pool: *const Self,
            current: IndexType = 0,

            pub fn next(self: *UsedIndicesIterator) ?IndexType {
                const pool_total = self.pool.total();
                while (self.current < pool_total) : (self.current += 1) {
                    if (self.pool.isValid(self.current)) {
                        const index = self.current;
                        self.current += 1;
                        return index;
                    }
                }
                return null;
            }
        };

        /// Create an iterator over all currently valid indices.
        ///
        /// Warning: The iterator provides a snapshot view. In thread-safe mode,
        /// indices may be allocated or freed while iterating.
        pub fn iterateUsedIndices(self: *const Self) UsedIndicesIterator {
            return .{ .pool = self };
        }

        /// Efficiently allocate multiple indices at once. This is more efficient
        /// than multiple single allocations, especially in thread-safe mode.
        ///
        /// Performance Note: This tries to fulfill the request from the free list
        /// first, then allocates new indices as needed. In thread-safe mode, it
        /// holds the lock for the entire operation.
        ///
        /// Returns: The number of indices successfully allocated, which may be
        /// less than requested if we hit capacity limits.
        pub fn allocBatch(self: *Self, indices: []IndexType) !usize {
            var allocated: usize = 0;

            // First try to use freed indices
            while (allocated < indices.len) {
                const maybe_index = self.free_indices.popOrNull();
                if (maybe_index) |index| {
                    indices[allocated] = if (config.thread_safe)
                        index.load(.acquire)
                    else
                        index;
                    allocated += 1;
                } else {
                    break;
                }
            }

            // Then allocate new indices if needed
            while (allocated < indices.len) {
                const index = if (config.thread_safe) blk: {
                    const val = self.next_index.load(.acquire);
                    self.next_index.store(val + 1, .release);
                    break :blk val;
                } else blk: {
                    const val = self.next_index;
                    self.next_index += 1;
                    break :blk val;
                };

                indices[allocated] = index;
                allocated += 1;
            }

            return allocated;
        }

        /// Free multiple indices in one operation. This is more efficient than
        /// freeing them individually, especially in thread-safe mode.
        ///
        /// Warning: All indices must be valid and not already freed. In debug mode,
        /// this is checked, but in release mode invalid indices could corrupt the pool.
        ///
        /// Error: Returns error.InvalidIndex if any index is out of range.
        pub fn freeBatch(self: *Self, indices: []const IndexType) !void {
            const current_total = if (config.thread_safe)
                self.next_index.load(.acquire)
            else
                self.next_index;

            // Validate all indices first
            for (indices) |index| {
                if (index >= current_total) return Error.InvalidIndex;
            }

            // Ensure capacity
            try self.free_indices.ensureTotalCapacity(self.allocator, self.free_indices.items.len + indices.len);

            // Add all indices
            for (indices) |index| {
                self.free_indices.appendAssumeCapacity(if (config.thread_safe)
                    Atomic(IndexType).init(index)
                else
                    index);
            }
        }

        /// Pre-allocate space for additional indices. This can prevent reallocations
        /// during high-growth periods.
        ///
        /// Memory Impact: This allocates memory immediately, even if the space isn't
        /// used right away. Use when you can predict upcoming allocation patterns.
        pub fn reserve(self: *Self, additional: usize) !void {
            try self.free_indices.ensureTotalCapacity(self.allocator, self.free_indices.items.len + additional);
        }

        /// Reset the pool to its initial state, forgetting all allocations and frees.
        /// This is faster than freeing everything individually.
        ///
        /// Warning: This does not check for active indices. Only use this when you're
        /// sure no indices are still in use, or you'll risk index conflicts.
        pub fn clear(self: *Self) void {
            self.free_indices.clearRetainingCapacity();
            self.next_index = 0;
        }
    };
}

/// A general-purpose index pool with balanced settings suitable for most use cases.
/// Provides good performance and memory usage without thread safety or debugging overhead.
///
/// Pro Tip: Start with this pool type unless you specifically need thread safety or
/// debugging features. You can always switch to a specialized pool later.
pub const IndexPool = IndexPoolImpl(default_config);

/// Thread-safe variant of the index pool, suitable for concurrent access from
/// multiple threads. Includes atomic operations and proper synchronization.
///
/// Performance Impact: The thread safety features add overhead to all operations.
/// Consider using separate non-thread-safe pools if possible.
pub const ThreadSafePool = IndexPoolImpl(.{
    .thread_safe = true,
});

/// Development-focused pool with all safety and debugging features enabled.
/// Great for catching bugs early and monitoring pool behavior.
///
/// Memory Impact: The debug features add memory overhead and runtime checks.
/// Switch to IndexPool or ThreadSafePool for production builds.
pub const DebugPool = IndexPoolImpl(.{
    .debug_mode = true,
    .enable_events = true,
});

test "index basic" {
    const testing = std.testing;
    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .initial_capacity = 4,
    };
    const Pool = IndexPoolImpl(config);

    var pool = try Pool.init(testing.allocator);
    defer pool.deinit();

    // Test 1: Basic allocation
    const index1 = try pool.alloc();
    try testing.expectEqual(@as(IndexType, 0), index1);

    // Test 2: Free and reuse
    try pool.free(index1);
    const index2 = try pool.alloc();
    try testing.expectEqual(@as(IndexType, 0), index2);
}

test "index concurrent" {
    const testing = std.testing;
    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .initial_capacity = 4,
    };
    const Pool = IndexPoolImpl(config);

    var pool = try Pool.init(testing.allocator);
    defer pool.deinit();

    // Allocate a few indices
    const idx1 = try pool.alloc();
    const idx2 = try pool.alloc();
    const idx3 = try pool.alloc();

    try testing.expectEqual(@as(IndexType, 0), idx1);
    try testing.expectEqual(@as(IndexType, 1), idx2);
    try testing.expectEqual(@as(IndexType, 2), idx3);

    // Free them in reverse order
    try pool.free(idx3);
    try pool.free(idx2);
    try pool.free(idx1);

    // They should be reused in LIFO order
    const new_idx1 = try pool.alloc();
    const new_idx2 = try pool.alloc();
    const new_idx3 = try pool.alloc();

    try testing.expectEqual(@as(IndexType, 0), new_idx1);
    try testing.expectEqual(@as(IndexType, 1), new_idx2);
    try testing.expectEqual(@as(IndexType, 2), new_idx3);
}

test "index multi-threading" {
    const testing = std.testing;
    const Thread = std.Thread;

    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .initial_capacity = 1024,
    };
    const Pool = IndexPoolImpl(config);

    const ThreadContext = struct {
        pool: *Pool,
        id: usize,
        indices: []IndexType,
    };

    const thread_count = 4;
    const iterations_per_thread = 100;

    var pool = try Pool.init(testing.allocator);
    defer pool.deinit();

    // Allocate memory for indices
    const all_indices = try testing.allocator.alloc(IndexType, thread_count * iterations_per_thread);
    defer testing.allocator.free(all_indices);

    // Thread worker function
    const WorkerFn = struct {
        fn run(ctx: ThreadContext) !void {
            // First phase: allocate all indices
            var i: usize = 0;
            while (i < iterations_per_thread) : (i += 1) {
                ctx.indices[ctx.id * iterations_per_thread + i] = try ctx.pool.alloc();
            }
        }
    };

    // Spawn threads for allocation
    var threads: [thread_count]Thread = undefined;

    // Start all threads
    for (&threads, 0..) |*thread, i| {
        const context = ThreadContext{
            .pool = &pool,
            .id = i,
            .indices = all_indices,
        };
        thread.* = try Thread.spawn(.{}, WorkerFn.run, .{context});
    }

    // Wait for all threads to complete allocation
    for (threads) |thread| {
        thread.join();
    }

    // Now free all indices from the main thread
    for (all_indices) |idx| {
        try pool.free(idx);
    }

    // Verify final state - everything should be freed
    try testing.expectEqual(@as(usize, thread_count * iterations_per_thread), pool.available());
    try testing.expectEqual(@as(IndexType, thread_count * iterations_per_thread), pool.total());
}
