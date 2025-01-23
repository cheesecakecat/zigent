//! Zigent Entity-Component-System
//! Copyright (c) 2024 Zigent Contributors
//! SPDX-License-Identifier: MIT OR Apache-2.0

const std = @import("std");
const testing = std.testing;

const generation = @import("entity/generation.zig");
const index = @import("entity/index.zig");
const entity = @import("entity.zig");

test {
    testing.refAllDecls(@This());
}
