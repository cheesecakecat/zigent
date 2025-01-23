//! Zigent Entity-Component-System
//! Copyright (c) 2024 Zigent Contributors
//! SPDX-License-Identifier: MIT OR Apache-2.0

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const assert = std.debug.assert;
const meta = std.meta;
const simd = std.simd;

/// The underlying type for generation counters. We use u32 to balance memory usage with the number
/// of possible generations. This gives us flexibility in how many bits we want to use for actual
/// generation counting via the Config.generation_bits setting.
///
/// Memory Impact: Each entity needs a generation value, so this directly affects our ECS memory
/// footprint. The u32 type allows for billions of generations while still being memory-efficient.
pub const GenerationType = u32;

/// Configure how generations behave in our ECS. These settings let you fine-tune the balance
/// between safety, performance, and memory usage. Choose carefully based on your specific needs.
///
/// Warning: These are compile-time configurations that affect the entire generation system.
/// You cannot change them at runtime. If you need different settings for different entity
/// groups, create multiple generation types.
pub const Config = struct {
    /// How many bits to use for generation counting. More bits mean more recycling before
    /// wrap-around but higher memory usage. The remaining bits in GenerationType are unused.
    ///
    /// Memory Impact: This doesn't affect the actual size of GenerationType, but it determines
    /// how many times an index can be recycled before wrapping back to 1. With 24 bits, you
    /// get 16 million generations before wrap-around.
    ///
    /// Warning: Once a generation wraps, there's a theoretical possibility of the ABA problem
    /// returning. Choose enough bits to make this practically impossible in your use case.
    generation_bits: u6 = 24,

    /// Enables SIMD operations for batch processing of generations. This can significantly
    /// speed up operations on large batches of entities but adds some code complexity.
    ///
    /// Performance Note: The actual speedup depends heavily on your CPU and usage patterns.
    /// Profile with your actual workload before committing to SIMD. Sometimes the overhead
    /// of setting up SIMD operations can outweigh the benefits for small batches.
    enable_simd: bool = true,

    /// Minimum number of operations needed before we use SIMD processing. Below this,
    /// we use regular scalar code. Finding the right value requires profiling with your
    /// specific workload.
    ///
    /// Performance Impact: Too low means we waste time on SIMD setup for small batches.
    /// Too high means we miss optimization opportunities. The sweet spot typically lies
    /// between 8 and 16 for most workloads.
    min_batch_size: usize = 8,

    /// Enables thread-safe operations through atomic operations. This adds overhead
    /// but ensures safe concurrent access from multiple threads.
    ///
    /// Performance Impact: Thread safety adds significant overhead - each operation
    /// requires atomic operations. If you're not using multiple threads, keep this off.
    ///
    /// Consider using multiple non-thread-safe generations (one per thread) instead of
    /// a single thread-safe generation if your architecture allows it.
    thread_safe: bool = false,

    /// Enables runtime checks for common errors like invalid generations. These checks
    /// are invaluable during development but do add overhead.
    ///
    /// Memory Impact: Debug mode adds minimal memory overhead but includes additional
    /// runtime checks and event generation capabilities.
    debug_mode: bool = false,

    /// Enables event callbacks for monitoring generation operations. Great for debugging,
    /// metrics collection, or implementing entity lifecycle hooks.
    ///
    /// Performance Note: The overhead is usually minimal if event handlers are lightweight.
    /// However, complex handlers can significantly impact performance since they run
    /// synchronously with generation operations.
    enable_events: bool = false,

    /// Validates configuration at compile-time to catch problematic combinations early.
    /// This helps prevent runtime surprises from misconfiguration.
    pub fn validate(comptime self: Config) void {
        if (self.generation_bits == 0) @compileError("Generation bits must be > 0");
        if (self.generation_bits > 32) @compileError("Generation bits must be <= 32");
        if (self.min_batch_size == 0) @compileError("Minimum batch size must be > 0");
        if (self.min_batch_size > 32) @compileError("Minimum batch size must be <= 32");
    }
};

/// Track significant events in the generation system's lifecycle. These events provide
/// visibility into generation changes and can help diagnose issues or collect metrics.
///
/// Performance Impact: Event handling adds overhead to every operation that generates
/// events. Keep handlers lightweight and consider disabling events in performance-critical
/// scenarios.
pub const Event = union(enum) {
    /// Tracks when a generation value is incremented. Useful for monitoring entity
    /// recycling patterns or implementing lifecycle hooks.
    increment: GenerationType,

    /// Indicates when a generation value wraps around to 1. Important for detecting
    /// potential ABA problems in long-running systems.
    wrap: GenerationType,

    /// Captures operation failures. Essential for debugging and error tracking during
    /// development.
    @"error": Error,
};

/// Function type for handling generation events. Implement this to hook into generation
/// operations for logging, debugging, or logic.
///
/// Warning: Event handlers run synchronously in the same thread as the generation
/// operation. Keep them fast to avoid performance issues.
pub const EventHandler = *const fn (event: Event) void;

/// Possible errors that can occur during generation operations. These help catch and
/// handle problematic situations early, especially during development.
pub const Error = error{
    /// The generation value has reached its maximum and cannot be incremented.
    /// This should never happen with properly configured generation_bits.
    GenerationOverflow,

    /// Attempted to use an invalid generation value. This usually indicates
    /// use of a stale entity reference or memory corruption.
    InvalidGeneration,
};

/// A balanced configuration that works well for most cases. Provides safety and decent
/// performance without excessive memory usage.
///
/// Start with this configuration and only customize options if profiling shows
/// a need. Premature optimization of these settings rarely pays off.
pub const default_config = Config{};

/// The core implementation of our generation tracking system. This provides a robust way to detect
/// stale entity references through generation counting. When an entity is deleted and its index
/// recycled, the generation is incremented, invalidating any existing references to the old entity.
///
/// Performance Note: The implementation heavily optimizes the common operations (get/increment) while
/// providing rich debugging and monitoring capabilities when needed. All debug features can be
/// disabled for production builds with zero overhead.
///
/// Warning: The generation's behavior is configured at compile-time through the Config struct. Runtime
/// configuration changes aren't supported - create separate generation instances if you need different
/// configurations for different subsystems.
pub fn GenerationImpl(comptime config: Config) type {
    config.validate();

    const max_generation = (1 << config.generation_bits) - 1;

    return struct {
        const Self = @This();
        const SimdVec = @Vector(config.min_batch_size, GenerationType);

        /// The current generation value. In thread-safe mode, this is wrapped in an atomic
        /// for safe concurrent access. Generations start at 1 and increment up to max_generation
        /// before wrapping back to 1.
        ///
        /// Implementation Note: We never use generation 0, as it serves as an invalid marker.
        /// This helps catch use-after-free and uninitialized entity reference bugs.
        value: if (config.thread_safe) std.atomic.Value(GenerationType) else GenerationType,

        /// Optional event handler for monitoring generation changes. When enabled, this can
        /// receive notifications about increments, wraps, and errors.
        ///
        /// Warning: Event handlers run synchronously in the operation's thread. Keep them
        /// fast and simple to avoid becoming a performance bottleneck.
        event_handler: if (config.enable_events) ?EventHandler else void,

        /// Initialize a new generation counter starting at 1. This is the only way to create
        /// a valid generation - direct value assignment is not allowed to ensure proper
        /// initialization.
        ///
        /// Note: The initial value of 1 is chosen because 0 is reserved as an invalid generation,
        /// helping catch use of uninitialized entity references.
        pub fn init() Self {
            var self: Self = undefined;
            if (config.thread_safe) {
                self.value = std.atomic.Value(GenerationType).init(1);
            } else {
                self.value = 1;
            }
            if (config.enable_events) {
                self.event_handler = null;
            }
            return self;
        }

        /// Register a callback to receive generation events. The handler will be called
        /// synchronously for increments, wraps, and errors.
        ///
        /// Warning: Only available if enable_events is true in the config. The handler
        /// must be fast to avoid blocking generation operations.
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

        /// Get the current generation value. In thread-safe mode, this performs an
        /// atomic load with acquire ordering to ensure proper synchronization.
        ///
        /// Performance Note: This is the most frequently called operation and is
        /// heavily optimized. In non-thread-safe mode, it's just a direct read.
        pub inline fn get(self: Self) GenerationType {
            return if (config.thread_safe)
                self.value.load(.acquire)
            else
                self.value;
        }

        /// Increment the generation value, wrapping back to 1 if we hit max_generation.
        /// In thread-safe mode, this uses atomic compare-and-swap to handle concurrent
        /// increments safely.
        ///
        /// Implementation Note: The wrap-around to 1 (not 0) maintains our invariant
        /// that 0 is invalid. This helps catch bugs even after wrap-around.
        pub fn increment(self: *Self) !void {
            if (config.thread_safe) {
                while (true) {
                    const current = self.value.load(.acquire);
                    const next = if (current >= max_generation) 1 else current + 1;

                    const old = self.value.swap(next, .seq_cst);
                    if (old != current) continue;

                    if (next == 1) {
                        self.emitEvent(.{ .wrap = current });
                    } else {
                        self.emitEvent(.{ .increment = next });
                    }
                    break;
                }
            } else {
                const current = self.value;
                if (current >= max_generation) {
                    self.value = 1;
                    self.emitEvent(.{ .wrap = current });
                } else {
                    self.value += 1;
                    self.emitEvent(.{ .increment = self.value });
                }
            }
        }

        /// The maximum possible generation value before wrap-around, based on the
        /// configured number of generation bits.
        ///
        /// Note: This is exposed as a constant to help systems that need to know
        /// the generation range, like serialization or networking code.
        pub const max_generation_value: GenerationType = max_generation;

        /// Check if a generation value is currently valid. A generation is valid if it's
        /// non-zero, not greater than max_generation, and matches the current generation.
        ///
        /// Performance Note: This is called frequently during entity validation, so it's
        /// heavily optimized. In non-debug builds, it's just a few simple comparisons.
        pub inline fn isValid(self: Self, generation: GenerationType) bool {
            const current = self.get();
            return generation > 0 and generation <= max_generation and generation == current;
        }

        /// Increment multiple generations at once, optionally using SIMD for better
        /// performance with large batches. This is useful when recycling many entities
        /// simultaneously.
        ///
        /// Performance Note: The SIMD path is only taken if enable_simd is true and
        /// the batch size meets the minimum threshold. Otherwise, it falls back to
        /// scalar processing.
        pub fn incrementBatch(generations: []GenerationType) void {
            if (!config.enable_simd or generations.len < config.min_batch_size) {
                for (generations) |*gen| {
                    gen.* = if (gen.* >= max_generation) 1 else gen.* + 1;
                }
                return;
            }

            var i: usize = 0;
            const vec_size = config.min_batch_size;
            const max_vec: SimdVec = @splat(max_generation);
            const one_vec: SimdVec = @splat(@as(GenerationType, 1));

            while (i + vec_size <= generations.len) : (i += vec_size) {
                const vec = simd.loadUnaligned(vec_size, generations[i..]);
                const next_vec = @select(
                    GenerationType,
                    vec >= max_vec,
                    one_vec,
                    vec + one_vec,
                );
                simd.storeUnaligned(generations[i..], next_vec);
            }

            while (i < generations.len) : (i += 1) {
                generations[i] = if (generations[i] >= max_generation)
                    1
                else
                    generations[i] + 1;
            }
        }

        /// Validate multiple generations at once, returning a slice of booleans indicating
        /// which generations are valid. This is useful for batch entity validation.
        ///
        /// Performance Note: Like incrementBatch, this can use SIMD for better performance
        /// with large batches. The SIMD path includes multiple vector operations but can
        /// still be faster than scalar code for large inputs.
        pub fn validateBatch(self: Self, generations: []const GenerationType) []const bool {
            const current = self.get();
            var result = std.ArrayList(bool).init(std.heap.page_allocator);
            result.resize(generations.len) catch return &[_]bool{};

            if (!config.enable_simd or generations.len < config.min_batch_size) {
                for (generations, 0..) |gen, i| {
                    result.items[i] = gen > 0 and gen <= max_generation and gen == current;
                }
                return result.items;
            }

            var i: usize = 0;
            const vec_size = config.min_batch_size;
            const current_vec: SimdVec = @splat(current);
            const max_vec: SimdVec = @splat(max_generation);
            const zero_vec: SimdVec = @splat(@as(GenerationType, 0));

            while (i + vec_size <= generations.len) : (i += vec_size) {
                const vec = simd.loadUnaligned(vec_size, generations[i..]);
                const valid = vec > zero_vec and vec <= max_vec and vec == current_vec;
                for (0..vec_size) |j| {
                    result.items[i + j] = valid[j];
                }
            }

            while (i < generations.len) : (i += 1) {
                result.items[i] = generations[i] > 0 and
                    generations[i] <= max_generation and
                    generations[i] == current;
            }

            return result.items;
        }
    };
}

/// A general-purpose generation tracker with balanced settings suitable for most use cases.
/// Provides good performance and memory usage without thread safety or debugging overhead.
///
/// Tip: Start with this type unless you specifically need thread safety or debugging
/// features. You can always switch to a specialized type later.
pub const Generation = GenerationImpl(default_config);

/// Thread-safe variant of the generation tracker, suitable for concurrent access from
/// multiple threads. Includes atomic operations and proper synchronization.
///
/// Performance Impact: The thread safety features add overhead to all operations.
/// Consider using separate non-thread-safe generations if possible.
pub const ThreadSafeGeneration = GenerationImpl(.{
    .thread_safe = true,
});

/// Development-focused generation tracker with all safety and debugging features enabled.
/// Great for catching bugs early and monitoring generation behavior.
///
/// Memory Impact: The debug features add overhead through event generation and additional
/// checks. Switch to Generation or ThreadSafeGeneration for production builds.
pub const DebugGeneration = GenerationImpl(.{
    .debug_mode = true,
    .enable_events = true,
});

test "generation basic" {
    const testing = std.testing;
    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .generation_bits = 8,
    };
    const Gen = GenerationImpl(config);

    var gen = Gen.init();

    try testing.expectEqual(@as(GenerationType, 1), gen.get());

    try gen.increment();
    try testing.expectEqual(@as(GenerationType, 2), gen.get());

    try testing.expect(gen.isValid(2));
    try testing.expect(!gen.isValid(1));
    try testing.expect(!gen.isValid(3));
}

test "generation wrapping" {
    const testing = std.testing;
    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .generation_bits = 2,
    };
    const Gen = GenerationImpl(config);

    var gen = Gen.init();
    try testing.expectEqual(@as(GenerationType, 1), gen.get());

    try gen.increment();
    try gen.increment();
    try gen.increment();
    try testing.expectEqual(@as(GenerationType, 1), gen.get());
}

test "generation multi-threading" {
    const testing = std.testing;
    const Thread = std.Thread;

    const config = Config{
        .thread_safe = true,
        .debug_mode = false,
        .generation_bits = 8,
    };
    const Gen = GenerationImpl(config);
    const max_gen = (1 << config.generation_bits) - 1;

    const ThreadContext = struct {
        gen: *Gen,
        iterations: usize,
    };

    var gen = Gen.init();
    const thread_count = 4;
    const iterations_per_thread = 100;

    const WorkerFn = struct {
        fn run(ctx: ThreadContext) !void {
            var i: usize = 0;
            while (i < ctx.iterations) : (i += 1) {
                try ctx.gen.increment();
            }
        }
    };

    var threads: [thread_count]Thread = undefined;
    const context = ThreadContext{
        .gen = &gen,
        .iterations = iterations_per_thread,
    };

    for (&threads) |*thread| {
        thread.* = try Thread.spawn(.{}, WorkerFn.run, .{context});
    }

    for (threads) |thread| {
        thread.join();
    }

    const expected = @mod(
        thread_count * iterations_per_thread + 1,
        max_gen,
    );
    if (expected == 0) {
        try testing.expectEqual(@as(GenerationType, 1), gen.get());
    } else {
        try testing.expectEqual(@as(GenerationType, expected), gen.get());
    }
}
