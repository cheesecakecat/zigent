const std = @import("std");
const testing = std.testing;

const generation = @import("entity/generation.zig");
const index = @import("entity/index.zig");

test {
    testing.refAllDecls(@This());
}
