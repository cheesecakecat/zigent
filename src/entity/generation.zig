// TODO: Document well

const std = @import("std");
const builtin = @import("builtin");
const math = std.math;
const assert = std.debug.assert;
const meta = std.meta;
const simd = std.simd;

pub const GenerationType = u32;

pub const Config = struct {
    generation_bits: u6 = 24,

    enable_simd: bool = true,

    min_batch_size: usize = 8,

    thread_safe: bool = false,

    debug_mode: bool = false,

    enable_events: bool = false,

    pub fn validate(comptime self: Config) void {
        if (self.generation_bits == 0) @compileError("Generation bits must be > 0");
        if (self.generation_bits > 32) @compileError("Generation bits must be <= 32");
        if (self.min_batch_size == 0) @compileError("Minimum batch size must be > 0");
        if (self.min_batch_size > 32) @compileError("Minimum batch size must be <= 32");
    }
};

pub const Event = union(enum) {
    increment: GenerationType,
    wrap: GenerationType,
    @"error": Error,
};

pub const EventHandler = *const fn (event: Event) void;

pub const Error = error{
    GenerationOverflow,
    InvalidGeneration,
};

pub const default_config = Config{};

pub fn GenerationImpl(comptime config: Config) type {
    config.validate();

    const max_generation = (1 << config.generation_bits) - 1;

    return struct {
        const Self = @This();
        const SimdVec = @Vector(config.min_batch_size, GenerationType);

        value: if (config.thread_safe) std.atomic.Value(GenerationType) else GenerationType,

        event_handler: if (config.enable_events) ?EventHandler else void,

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

        pub fn setEventHandler(self: *Self, handler: EventHandler) void {
            if (config.enable_events) {
                self.event_handler = handler;
            }
        }

        inline fn emitEvent(self: *Self, event: Event) void {
            if (config.enable_events) {
                if (self.event_handler) |handler| {
                    handler(event);
                }
            }
        }

        pub inline fn get(self: Self) GenerationType {
            return if (config.thread_safe)
                self.value.load(.acquire)
            else
                self.value;
        }

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

        pub const max_generation_value: GenerationType = max_generation;

        pub inline fn isValid(self: Self, generation: GenerationType) bool {
            const current = self.get();
            return generation > 0 and generation <= max_generation and generation == current;
        }

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

pub const Generation = GenerationImpl(default_config);

pub const ThreadSafeGeneration = GenerationImpl(.{
    .thread_safe = true,
});

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
