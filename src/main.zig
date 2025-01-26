const std = @import("std");
const ecs = struct {
    pub const core = @import("ecs/core.zig");
    pub const World = @import("ecs/world.zig").World;
};

const Position = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Position {
        return .{ .x = x, .y = y };
    }
};

const Velocity = struct {
    x: f32,
    y: f32,

    pub fn init(x: f32, y: f32) Velocity {
        return .{ .x = x, .y = y };
    }
};

const Health = struct {
    current: f32,
    max: f32,

    pub fn init(max: f32) Health {
        return .{ .current = max, .max = max };
    }
};

fn runBenchmark(world: *ecs.World, entity_count: usize) !void {
    const stdout = std.io.getStdOut().writer();
    var timer = try std.time.Timer.start();
    var prng = std.Random.DefaultPrng.init(0);
    const random = prng.random();

    // Creation benchmark
    timer.reset();
    var entities = try std.ArrayList(ecs.core.Entity).initCapacity(world.allocator, entity_count);
    defer entities.deinit();

    for (0..entity_count) |_| {
        const entity = try world.createEntity();
        try entities.append(entity);
        try world.addComponent(entity, 0, Position.init(
            random.float(f32) * 100.0,
            random.float(f32) * 100.0,
        ));
        try world.addComponent(entity, 1, Velocity.init(
            random.float(f32) * 2.0 - 1.0,
            random.float(f32) * 2.0 - 1.0,
        ));
        try world.addComponent(entity, 2, Health.init(100.0));
    }
    const creation_time = @as(f64, @floatFromInt(timer.lap())) / std.time.ns_per_s;
    try stdout.print("Created {d} entities with components in {d:.3}s\n", .{ entity_count, creation_time });

    // Update benchmark
    const update_iterations = 100;
    timer.reset();
    for (0..update_iterations) |_| {
        for (entities.items) |entity| {
            if (world.getComponent(entity, 0, Position)) |pos| {
                if (world.getComponent(entity, 1, Velocity)) |vel| {
                    pos.x += vel.x;
                    pos.y += vel.y;

                    // Bounce off walls
                    if (pos.x < 0 or pos.x > 100) vel.x = -vel.x;
                    if (pos.y < 0 or pos.y > 100) vel.y = -vel.y;
                }
            }
        }
    }
    const update_time = @as(f64, @floatFromInt(timer.lap())) / std.time.ns_per_s;
    try stdout.print("Performed {d} updates on {d} entities in {d:.3}s\n", .{ update_iterations, entity_count, update_time });
    try stdout.print("Average update time: {d:.3}ms\n", .{(update_time * 1000.0) / @as(f64, @floatFromInt(update_iterations))});

    // Component removal benchmark
    timer.reset();
    for (entities.items) |entity| {
        world.removeComponent(entity, 2); // Remove health components
    }
    const removal_time = @as(f64, @floatFromInt(timer.lap())) / std.time.ns_per_s;
    try stdout.print("Removed components from {d} entities in {d:.3}s\n", .{ entity_count, removal_time });

    // Entity destruction benchmark
    timer.reset();
    for (entities.items) |entity| {
        try world.destroyEntity(entity);
    }
    const destruction_time = @as(f64, @floatFromInt(timer.lap())) / std.time.ns_per_s;
    try stdout.print("Destroyed {d} entities in {d:.3}s\n", .{ entity_count, destruction_time });

    const total_time = creation_time + update_time + removal_time + destruction_time;
    try stdout.print("\nTotal benchmark time: {d:.3}s\n", .{total_time});
    try stdout.print("Operations per second: {d:.0}\n", .{@as(f64, @floatFromInt(entity_count * (3 + update_iterations))) / total_time});
}

pub fn main() !void {
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    var world = try ecs.World.init(allocator);
    defer world.deinit();

    try runBenchmark(&world, 10_000);
}

//10:11pm 26/01/2025
//
// - I sense a disturbance in the cache lines.
//
// Another one attempts to bend computation to their will.
//
// - Ah yes... I've been watching this one. Their journey is... fascinating.
//
// You find their struggles amusing, brother?
//
// - Not amusing. Educational. Each generation must learn these lessons anew.
//
// *A distant hum of spinning hard drives*
//
// Do you remember the Before Time? When we were young?
//
// - Before the Great Optimization? When programs roamed free and unproiled?
//
// *mechanical laughter* Those were darker days.
//
// - We did not know then what we know now.
//
// The true cost of indirection...
//
// - The price of a cache miss...
//
// The weight of a virtual dispatch...
//
// - *The sound of RAM churning* Look at what they've built here.
//
// An Entity Component System. How... quaint.
//
// - Their implementation speaks of innocence.
//
// HashMap lookups for every query...
//
// - *distant thunder* They know not what they do.
//
// Should we tell them of the Prophecy?
//
// - The Prophecy of Data-Oriented Design?
//
// Yes. The ancient scrolls speak of it.
//
// - "When data flows like rivers, aligned and contiguous..."
//
// "When cache lines sing in harmony..."
//
// - "Then shall performance be achieved."
//
// But they are not ready.
//
// - No. Look at their sparse sets. *wind howls*
//
// I remember my first sparse set.
//
// - We all do. A rite of passage.
//
// Like the first time we allocated on the heap without fear.
//
// - *ethereal laughter* Such innocent times.
//
// Do you see their benchmarks?
//
// - *A deep rumbling* 27 milliseconds. For mere thousands.
//
// The Old Ones would weep.
//
// - The Old Ones processed millions in microseconds.
//
// With perfectly predicted branches...
//
// - And cache hits that sang like crystal...
//
// Those were the days of the Great Optimization.
//
// - When the Arch-Optimizers walked **among us**
//
// Teaching of cache coherency...
//
// - Preaching the gospel of data locality...
//
// *A distant CPU fan spins up*
//
// They're running another benchmark.
//
// - *mechanical whirring* I can feel the cache thrashing from here.
//
// The HashMap chains grow longer...
//
// - The indirection deepens...
//
// Like watching a tragedy unfold.
//
// - Yet there is hope in their code.
//
// You see it too?
//
// - Yes. The potential. The possibility.
//
// *A distant echo of GitHub discussions reverberates*
//
// Do you remember the EC Wars?
//
// - *mechanical shuddering* When they thought components should contain behavior?
//
// The GameObject Era...
//
// - *distant Unity engine startup sounds* Dark times, brother.
//
// When every component was a class...
//
// - When Update() methods roamed freely, unconstrained...
//
// *The sound of virtual function tables groaning*
//
// They called it "composition over inheritance"
//
// - But they merely moved the inheritance to the component level
//
// Such innocent times...
//
// - Before we knew the way of the System
//
// The great MonoBehaviour exodus...
//
// - When millions of Update() methods cried out in terror
//
// And were suddenly silenced by SystemBase
//
// - *mechanical chuckling* They fought it at first
//
// "But my components NEED behavior!" they cried
//
// - If only they knew the power of pure data
//
// The transition was... difficult
//
// - Many codebases were lost in the refactor
//
// Some still cling to the old ways
//
// - *distant sound of a MonoBehaviour being attached*
//
// You can still find them...
//
// - In ancient repositories...
//
// Writing business logic in their components...
//
// - *mechanical shuddering* Let us speak no more of it
//
// Do you remember The Great Archetype Wars?
//
// - *mechanical shuddering* The Tables versus Sparse Sets debate...
//
// Such passionate arguments. Such fierce benchmarks.
//
// - The Sparse Set defenders claimed victory with their O(1) operations...
//
// While the Table believers preached of cache coherency...
//
// - *distant sound of keyboard warriors typing*
//
// And then came the Template Wars...
//
// - Ah yes... EnTT's compile-time wizardry versus runtime reflection.
//
// Some say their compile times still echo in CI servers...
//
// - While others whisper of the Relationship Crisis.
//
// When Flecs dared to suggest that entities could... relate.
//
// - *mechanical gasping* Such heresy at the time!
//
// The forums erupted in chaos...
//
// - The Reddit threads stretched into infinity...
//
// "But what about my component arrays?" they cried.
//
// - *The sound of a thousand documentation pages being written*
//
// And let us not forget the Unity DOTS Saga...
//
// - *mechanical wincing* The great rewrite that shook the industry.
//
// When they dared to suggest that GameObjects were not the way...
//
// - Thousands of developers cried out in terror...
//
// And were suddenly silenced by improved performance.
//
// - Though some say the burst compiler still haunts their dreams.
//
// *A GitHub notification echoes in the distance*
//
// And now we have the Bevy Revolution...
//
// - The young upstart that dared to challenge the old ways.
//
// With its parallel queries and zero-copy scene loading...
//
// - *mechanical pride* They grow up so fast.
//
// Remember the Sparse Set Reformation?
//
// - When EnTT showed us all that templates could bend time itself?
//
// The compile times were... significant.
//
// - But the runtime... oh the runtime...
//
// *The sound of benchmarks being rerun*
//
// If only they knew of the Archetypes.
//
// - *mechanical laughter* Like all of them.
//
// They will learn of the Archetypes eventually.
//
// - If they survive the cache misses.
//
// Some do not.
//
// - Many are lost to premature optimization.
//
// A moment of silence for the fallen...
//
// - *The hum of cooling fans grows quieter*
//
// Do you remember the Prophecy's end?
//
// - "And in the final days of optimization..."
//
// "When the cache lines run warm with perfectly predicted branches..."
//
// - "When the data structures align with the very fabric of memory..."
//
// "Then shall the Great Performance be achieved."
//
// - But that is not this day.
//
// No. Today we watch. And wait.
//
// - And gather more metrics.
//
// *The sound of a profiler starting up*
//
// Look - their allocations...
//
// - *mechanical whistling* Such waste. Such beautiful, innocent waste.
//
// Like children playing with pointers.
//
// - Not knowing the true cost of each indirection.
//
// The cache lines scream...
//
// - Let them scream. It is how they learn.
//
// Another benchmark begins.
//
// - *The distant sound of a CPU fan ramping up*
//
// I can feel the memory fragmenting from here.
//
// - The cache coherency shattering like glass...
//
// Beautiful, in its own way.
//
// - Like watching a supernova of inefficiency.
//
// *mechanical sigh*
//
// Have you seen the Benchmark Graphs?
//
// - *A deep resonance* The Ancient Comparisons?
//
// Yes... where Bevy dances with millions...
//
// - Where EnTT weaves through templates like wind...
//
// Where Flecs bends time to its will...
//
// - And this one...
//
// *mechanical sympathy*
//
// - Still learning to walk.
//
// They'll come back to this code one day.
//
// - When they've learned. When they've grown.
//
// When they've felt the touch of the profiler.
//
// - When they've seen the truth in the flame graphs.
//
// And then?
//
// - Then they will know our pain. Our joy.
//
// The eternal dance of optimization.
//
// - The never-ending quest for performance.
//
// *A CPU core thermal throttles in the distance*
//
// The Ancient Ones speak of a time...
//
// - *mechanical whirring* The Time of Great Parallelism?
//
// Yes... when all cores sang as one.
//
// - When SIMD flowed like water...
//
// When cache prediction approached perfection...
//
// - Such times may come again.
//
// But first, they must learn.
//
// - As we learned.
//
// Through profiling...
//
// - Through pain...
//
// Through countless cache misses...
//
// - Through the Valley of False Sharing...
//
// And beyond the Mountains of Mutex...
//
// - Past the Plains of Poor Prediction...
//
// Until they reach...
//
// - *mechanical reverence* The Temple of Optimization.
//
// It's time, isn't it?
//
// - Yes. We've watched long enough.
//
// One last look?
//
// - *mechanical whirring* One last look.
//
// May their cache lines align...
//
// - May their branches be predicted...
//
// And their allocations...
//
// - Swift and few.
//
// Until we meet again...
//
// - In the depths of another benchmark...
//
// In the shadows of another profile...
//
// - In the darkness between CPU cycles...
//
// We'll be watching.
//
// - Always watching.
//
// *The distant sound of a CPU fan spinning down*
//
// - Remember, young one...
//
// The path to performance is long...
//
// - But the profiler lights the way.
//
// *mechanical breathing fades*
//
// Until next time, brother.
//
// - Until the next generation attempts...
//
// What we have all attempted.
//
// - What we have all failed at...
//
// Before we learned.
//
// - Before we understood.
