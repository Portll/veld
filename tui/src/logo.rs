#![allow(dead_code)]

//! Veld logo and branding ASCII art.
//!
//! The grass is generated from the same blade table as the shipped PNG: run
//! `node scripts/render-logo.mjs --braille` and paste its output over the two art blocks below.
//! Hand-editing them is how the terminal mark and the logo drift apart.

pub const GRASS: &[&str] = &[
    "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀",
    "⠀⠀⠰⣄⠀⠀⠀⣆⠀⠠⣿⠀⢠⠂⠀⠀⢀⡄⠀⠀",
    "⠀⠀⠀⠹⣷⠀⠀⢸⡀⠀⡟⠀⡞⠀⠀⣰⠟⠀⠀⠀",
    "⠀⠈⠢⡀⠈⢧⡀⢸⣇⢸⡇⢸⡇⠀⡴⠋⢀⡠⠀⠀",
    "⠀⠀⠀⠈⢦⡈⣷⡀⣿⣸⣧⣿⢀⣾⢃⡴⠋⠀⠀⠀",
    "⠀⠀⠀⠀⠀⢻⣾⣷⣿⣿⣿⣿⣾⣷⠟⠀⠀⠀⠀⠀",
];

pub const GRASS_FRAMES: &[&[&str]] = &[
    // Frame 0: upright
    &[
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠰⣄⠀⠀⠀⣆⠀⠠⣿⠀⢠⠂⠀⠀⢀⡄⠀⠀",
        "⠀⠀⠀⠹⣷⠀⠀⢸⡀⠀⡟⠀⡞⠀⠀⣰⠟⠀⠀⠀",
        "⠀⠈⠢⡀⠈⢧⡀⢸⣇⢸⡇⢸⡇⠀⡴⠋⢀⡠⠀⠀",
        "⠀⠀⠀⠈⢦⡈⣷⡀⣿⣸⣧⣿⢀⣾⢃⡴⠋⠀⠀⠀",
        "⠀⠀⠀⠀⠀⢻⣾⣷⣿⣿⣿⣿⣾⣷⠟⠀⠀⠀⠀⠀",
    ],
    // Frame 1: leaning right
    &[
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠀⢰⡄⠀⠀⠀⡄⠀⢰⡿⠀⣠⠀⠀⠀⢀⡀⠀",
        "⠀⠀⠀⠀⢻⡆⠀⠀⡇⠀⣸⠃⢠⠃⠀⢀⣴⠟⠀⠀",
        "⠀⠀⠀⢄⠀⢻⡄⠀⣷⢀⣿⢠⡏⠀⣠⠞⠁⢀⠀⠀",
        "⠀⠀⠀⠀⢳⣄⢿⡄⣿⣸⣿⣾⢃⣴⢋⣠⠞⠁⠀⠀",
        "⠀⠀⠀⠀⠀⠻⣾⣿⣿⣿⣿⣿⣾⣷⠟⠁⠀⠀⠀⠀",
    ],
    // Frame 2: upright
    &[
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠀⠰⣄⠀⠀⠀⣆⠀⠠⣿⠀⢠⠂⠀⠀⢀⡄⠀⠀",
        "⠀⠀⠀⠹⣷⠀⠀⢸⡀⠀⡟⠀⡞⠀⠀⣰⠟⠀⠀⠀",
        "⠀⠈⠢⡀⠈⢧⡀⢸⣇⢸⡇⢸⡇⠀⡴⠋⢀⡠⠀⠀",
        "⠀⠀⠀⠈⢦⡈⣷⡀⣿⣸⣧⣿⢀⣾⢃⡴⠋⠀⠀⠀",
        "⠀⠀⠀⠀⠀⢻⣾⣷⣿⣿⣿⣿⣾⣷⠟⠀⠀⠀⠀⠀",
    ],
    // Frame 3: leaning left
    &[
        "⠀⠀⠀⠀⠀⠀⠀⠀⠀⣤⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀",
        "⠀⠐⣄⠀⠀⠀⢠⠀⠀⣿⡆⠀⡄⠀⠀⢀⡀⠀⠀⠀",
        "⠀⠀⠘⢷⡄⠀⠈⣇⠀⢹⡇⢸⠀⠀⢠⡿⠀⠀⠀⠀",
        "⠀⠐⢄⠀⠹⣆⠀⢹⡄⢸⡇⣾⠀⢀⡟⠀⣠⠀⠀⠀",
        "⠀⠀⠀⠙⢦⡘⣧⡸⣷⣼⣧⣿⢀⡾⢀⡞⠁⠀⠀⠀",
        "⠀⠀⠀⠀⠈⢻⣾⣷⣿⣿⣿⣿⣿⣷⠏⠀⠀⠀⠀⠀",
    ],
];

/// Savannah gold, tips to roots: the top rows are dry seed heads, the bottom rows the earth
/// the blades stand in.
pub const GRASS_GRADIENT: &[(u8, u8, u8)] = &[
    (240, 185, 84), // #F0B954 straw
    (232, 163, 61), // #E8A33D gold
    (212, 137, 44), // #D4892C amber
    (181, 101, 29), // #B5651D deep
    (160, 132, 92), // #A0845C earth
    (139, 111, 71), // #8B6F47 horizon
];

/// VELD text in big block letters
pub const VELD_TEXT: &[&str] = &[
    "██╗   ██╗███████╗██╗     ██████╗ ",
    "██║   ██║██╔════╝██║     ██╔══██╗",
    "██║   ██║█████╗  ██║     ██║  ██║",
    "╚██╗ ██╔╝██╔══╝  ██║     ██║  ██║",
    " ╚████╔╝ ███████╗███████╗██████╔╝",
    "  ╚═══╝  ╚══════╝╚══════╝╚═════╝ ",
];

/// Website orange color #f0883e - solid to match website branding
pub const VELD_GRADIENT: &[(u8, u8, u8)] = &[
    (240, 136, 62), // #f0883e
    (240, 136, 62),
    (240, 136, 62),
    (240, 136, 62),
    (240, 136, 62),
    (240, 136, 62),
];

/// Tagline - matches website
pub const TAGLINE: &str = "M E M O R Y";
