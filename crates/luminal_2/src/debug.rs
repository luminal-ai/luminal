use std::f32::consts::{FRAC_PI_2, TAU};

use eframe::{
    egui,
    egui::{Color32, Pos2, Vec2},
};
use egglog::Term;
use egui::ViewportId;
use itertools::Itertools;
use luminal::{
    prelude::petgraph::{Directed, Direction, graph::NodeIndex, prelude::StableGraph},
    shape::Expression,
};
use rustc_hash::{FxHashMap, FxHashSet};

use crate::{GraphTerm, Kernel};

#[derive(Debug, Clone, Copy)]
pub enum NodeShape {
    Circle,
    Triangle,
    InvertedTriangle,
    Square,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DisplayMode {
    TermType,
    LoopLevel,
}

pub type DisplayNode = (String, egui::Color32, NodeShape, Option<usize>);
pub type DisplayGraph = StableGraph<DisplayNode, ()>;

pub fn display_graph(graph: &StableGraph<impl TermToString, impl EdgeToString, Directed, u32>) {
    display_multiple_graphs(&[graph]);
}

pub fn display_multiple_graphs<T: TermToString, E: EdgeToString>(
    graphs: &[&StableGraph<T, E, Directed, u32>],
) {
    // Build independent display graphs/views
    let views: Vec<View> = graphs
        .iter()
        .map(|g| View::new(to_display_graph(g, &[])))
        .collect();

    let icon = eframe::icon_data::from_png_bytes(include_bytes!("../debugger_icon.png"))
        .expect("valid PNG");
    let _ = eframe::run_native(
        "Luminal Debugger",
        eframe::NativeOptions {
            viewport: egui::ViewportBuilder::default()
                .with_inner_size([if graphs.len() == 1 { 1000.0 } else { 2000.0 }, 1200.0])
                .with_decorations(false)
                .with_transparent(true)
                .with_close_button(true)
                .with_icon(icon)
                .with_title("Luminal Debugger"),
            ..Default::default()
        },
        Box::new(move |_cc| Ok(Box::new(Debugger::new(views)))),
    );
}

fn to_display_graph<T: TermToString, E: EdgeToString>(
    graph: &StableGraph<T, E, Directed, u32>,
    mark_nodes: &[(NodeIndex, Color32)],
) -> DisplayGraph {
    // 1) copy nodes/edges
    let mut dg = DisplayGraph::new();
    let mut map = FxHashMap::default();
    for node in graph.node_indices() {
        let (r, c, b) = graph.node_weight(node).unwrap().term_to_string();
        map.insert(
            node,
            dg.add_node(
                if let Some((_, color)) = mark_nodes.iter().find(|(i, _)| *i == node) {
                    (r, *color, b, None)
                } else {
                    (r, c, b, None)
                },
            ),
        );
    }
    for e in graph.edge_indices() {
        let (u, v) = graph.edge_endpoints(e).unwrap();
        dg.add_edge(map[&u], map[&v], ());
    }

    // 2) derive loop levels
    let mut seen = dg
        .externals(Direction::Incoming)
        .chain(dg.externals(Direction::Outgoing))
        .collect::<FxHashSet<_>>();
    for n in &seen {
        if dg[*n].0.contains("LoopOut") {
            dg.node_weight_mut(*n).unwrap().3 = Some(0);
        }
    }

    let mut dfs = seen
        .iter()
        .flat_map(|n| dg.neighbors_directed(*n, Direction::Incoming))
        .collect_vec();

    while let Some(n) = dfs.pop() {
        if seen.contains(&n) {
            continue;
        }
        seen.insert(n);
        let curr_term = dg[n].0.clone();

        if let Some(outgoing_neighbor) = dg
            .neighbors_directed(n, Direction::Outgoing)
            .find(|x| seen.contains(x))
        {
            let (neighbor_weight, _, _, mut neighbor_levels) = dg[outgoing_neighbor].clone();
            if neighbor_weight.contains("LoopOut") {
                neighbor_levels = Some(neighbor_levels.unwrap_or_default() + 1);
            }
            if curr_term.contains("LoopIn") {
                neighbor_levels = neighbor_levels.map(|i| i - 1);
            }
            dg.node_weight_mut(n).unwrap().3 = neighbor_levels;
        } else if let Some(incoming_neighbor) = dg
            .neighbors_directed(n, Direction::Incoming)
            .find(|x| seen.contains(x))
        {
            let (neighbor_weight, _, _, mut neighbor_levels) = dg[incoming_neighbor].clone();
            if neighbor_weight.contains("LoopIn") {
                neighbor_levels = Some(neighbor_levels.unwrap_or_default() + 1);
            }
            if curr_term.contains("LoopOut") {
                neighbor_levels = neighbor_levels.map(|i| i - 1);
            }
            dg.node_weight_mut(n).unwrap().3 = neighbor_levels;
        }
        dfs.extend(dg.neighbors_undirected(n));
    }
    for (w, _, _, l) in dg.node_weights_mut() {
        if !w.contains("LoopIn") && !w.contains("LoopOut") {
            *l = None;
        }
    }

    dg
}

pub struct View {
    g: DisplayGraph,
    pos: Vec<Pos2>,
    dragging: Option<usize>,
    cam: Camera,
    need_fit: bool,
}

impl View {
    pub fn new(g: DisplayGraph) -> Self {
        Self {
            pos: layered_layout(&g, 400.0, 120.0, 100.0),
            g,
            dragging: None,
            cam: Camera::new(),
            need_fit: true,
        }
    }
}

pub struct Debugger {
    views: Vec<View>,
    should_close: bool,
}

impl Debugger {
    pub fn new(views: Vec<View>) -> Self {
        Self {
            views,
            should_close: false,
        }
    }
}

impl Debugger {
    fn setup_visuals_and_background(&self, ctx: &egui::Context) {
        // Make panels/windows transparent so rounded bg shows through
        let mut visuals = egui::Visuals::dark();
        visuals.panel_fill = Color32::TRANSPARENT;
        visuals.window_fill = Color32::TRANSPARENT;
        ctx.set_visuals(visuals);

        // Rounded background behind everything
        let screen = ctx.screen_rect();
        let rounding = egui::Rounding::same(9.0); // corner radius
        let bg = Color32::from_hex("#0D180A").unwrap();
        let bg_painter = ctx.layer_painter(egui::LayerId::background());
        // shrink a hair to avoid clipping the antialiased edge
        bg_painter.rect_filled(screen.shrink(0.5), rounding, bg);
    }

    fn handle_global_shortcuts(&mut self, ctx: &egui::Context) {
        // Global 'D' toggles display mode for all views
        if ctx.input(|i| i.key_pressed(egui::Key::D)) {
            for v in &mut self.views {
                v.cam.toggle_mode();
            }
            ctx.request_repaint();
        }

        // quit on Ctrl+Q
        if ctx.input(|i| i.key_pressed(egui::Key::Q) && (i.modifiers.ctrl || i.modifiers.command)) {
            self.should_close = true;
        }

        // Handle close requests
        if ctx.input(|i| i.viewport().close_requested()) {
            self.should_close = true;
        }

        if self.should_close {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
        }
    }

    fn draw_title_buttons(
        &mut self,
        ui: &mut egui::Ui,
        ctx: &egui::Context,
        painter: &egui::Painter,
        rect: egui::Rect,
        pad: f32,
    ) {
        // --- mac "traffic lights" ---
        let r = 6.0;
        let gap = 8.0;
        let cy = rect.center().y;
        let cx0 = rect.left() + pad + r;

        let centers = [
            Pos2::new(cx0 + 0.0 * (2.0 * r + gap), cy), // close
            Pos2::new(cx0 + 1.0 * (2.0 * r + gap), cy), // minimize
            Pos2::new(cx0 + 2.0 * (2.0 * r + gap), cy), // fullscreen
        ];
        let colors = [
            Color32::from_rgb(255, 95, 86),  // red
            Color32::from_rgb(255, 189, 46), // yellow
            Color32::from_rgb(39, 201, 63),  // green
        ];

        for (i, c) in centers.iter().enumerate() {
            // base circle
            painter.circle_filled(*c, r, colors[i]);

            // hit area & interaction
            let hit = egui::Rect::from_center_size(*c, egui::vec2(2.0 * r + 6.0, 2.0 * r + 6.0));
            let id = ui.make_persistent_id(("title_btn", i));
            let resp = ui
                .interact(hit, id, egui::Sense::click())
                .on_hover_cursor(egui::CursorIcon::PointingHand);

            // mac-style icon appears on hover (animated)
            let t = ui.ctx().animate_bool(id.with("hover"), resp.hovered()); // 0..1
            let icon_alpha = ((t * 255.0).round() as u8).min(255);
            let icon_color = Color32::from_white_alpha(icon_alpha);

            // optional hover ring
            if t > 0.0 {
                painter.circle_stroke(*c, r + 1.5, egui::Stroke::new(1.0 * t, icon_color));
            }

            // pick the glyph like macOS
            let glyph = match i {
                0 => "×", // close
                1 => "–", // minimize
                2 => "⤢", // fullscreen (diagonal arrows)
                _ => "",
            };

            // draw glyph centered in the circle
            painter.text(
                *c,
                egui::Align2::CENTER_CENTER,
                glyph,
                egui::FontId::proportional(10.0),
                icon_color,
            );

            // click behavior
            if resp.clicked() {
                match i {
                    0 => {
                        // hide instantly
                        ctx.send_viewport_cmd_to(
                            ViewportId::ROOT,
                            egui::ViewportCommand::Visible(false),
                        );
                        // then actually close on the next pump
                        ctx.send_viewport_cmd_to(ViewportId::ROOT, egui::ViewportCommand::Close);
                        // ensure we process it ASAP
                        ctx.request_repaint();
                    }
                    1 => ctx.send_viewport_cmd(egui::ViewportCommand::Minimized(true)),
                    2 => ctx.send_viewport_cmd(egui::ViewportCommand::Fullscreen(
                        !ctx.input(|inp| inp.viewport().fullscreen.unwrap_or(false)),
                    )),
                    _ => {}
                }
            }
        }
    }

    fn draw_title_bar(&mut self, ctx: &egui::Context) {
        const BAR_H: f32 = 36.0;
        const PAD: f32 = 10.0;

        egui::TopBottomPanel::top("custom_title_bar")
            .exact_height(BAR_H)
            .frame(egui::Frame::none())
            .show(ctx, |ui| {
                let mut rect = ui.max_rect();
                // avoid painting outside the rounded window edge
                rect = rect.shrink(0.5);

                let painter = ui.painter_at(rect);
                let top_round = egui::Rounding {
                    nw: 9.0,
                    ne: 9.0,
                    se: 0.0,
                    sw: 0.0,
                };

                // single fill with rounded top corners
                painter.rect_filled(rect, top_round, Color32::from_hex("#7FEE640D").unwrap());

                self.draw_title_buttons(ui, ctx, &painter, rect, PAD);

                // --- drag anywhere on the bar except over the buttons ---
                let r = 6.0;
                let gap = 8.0;
                let drag_rect = rect.shrink2(egui::vec2(PAD + 3.0 * (2.0 * r + gap), 0.0));
                let drag_resp = ui.interact(
                    drag_rect,
                    ui.make_persistent_id("title_drag"),
                    egui::Sense::drag(),
                );
                if drag_resp.drag_started()
                    || (drag_resp.is_pointer_button_down_on() && drag_resp.hovered())
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::StartDrag);
                }

                // centered title
                let title = "Luminal Debugger";
                painter.text(
                    Pos2::new(rect.center().x, rect.center().y),
                    egui::Align2::CENTER_CENTER,
                    title,
                    egui::FontId::new(14.0, egui::FontFamily::Monospace),
                    Color32::from_hex("#7FEE64").unwrap(),
                );

                // bottom border line
                painter.line_segment(
                    [
                        Pos2::new(rect.left(), rect.bottom() - 1.5),
                        Pos2::new(rect.right(), rect.bottom() - 1.5),
                    ],
                    egui::Stroke::new(3.0, Color32::from_hex("#7FEE64").unwrap()),
                );
            });
    }

    fn draw_top_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::top("top_multi").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.label("Pan: drag/bg scroll • Zoom: ⌘/Ctrl+wheel or pinch • Toggle mode: D");
                if ui.button("Fit all").clicked() {
                    for v in &mut self.views {
                        v.need_fit = true;
                    }
                }
                ui.separator();
                for (i, v) in self.views.iter().enumerate() {
                    ui.label(format!(
                        "View {i}: {:.1}x {:?}",
                        v.cam.zoom, v.cam.display_mode
                    ));
                }
            });
        });
    }

    fn handle_view_input(
        view: &mut View,
        ui_col: &mut egui::Ui,
        resp: &egui::Response,
        origin: Pos2,
    ) {
        // Read input (global), then gate by hovered/active
        let (mods, raw_scroll, pinch, pointer, drag_delta) = ui_col.input(|inp| {
            (
                inp.modifiers,
                inp.raw_scroll_delta,
                inp.zoom_delta_2d(),
                inp.pointer.clone(),
                inp.pointer.delta(),
            )
        });
        let hovered = resp.hovered();
        let active = hovered || view.dragging.is_some();

        // Zoom: only when hovered/active and cursor inside this rect
        if active {
            let do_pinch = (pinch.y - 1.0).abs() > 1e-3 && hovered;
            let do_wheel_zoom = hovered && (mods.command || mods.ctrl) && raw_scroll.y != 0.0;

            if do_pinch || do_wheel_zoom {
                if let Some(cursor) = pointer.hover_pos() {
                    if resp.rect.contains(cursor) {
                        let factor = if do_pinch {
                            pinch.y
                        } else {
                            (raw_scroll.y * -0.001).exp()
                        };
                        view.cam.zoom_at(origin, cursor, factor);
                        ui_col.ctx().request_repaint();
                    }
                }
            } else if hovered && raw_scroll != Vec2::ZERO {
                // Trackpad/mouse scroll pans only when hovered
                view.cam.pan += raw_scroll;
            }
        }

        // Dragging: start only on this column; continue while pressed even if cursor leaves
        if active {
            if let Some(pos) = pointer.interact_pos() {
                let pos_in_this = resp.rect.contains(pos);

                if hovered && pointer.primary_pressed() && view.dragging.is_none() {
                    let pick_r2 = 16.0 * 16.0;
                    view.dragging = view
                        .pos
                        .iter()
                        .enumerate()
                        .map(|(j, &w)| (j, view.cam.w2s(origin, w).distance_sq(pos)))
                        .filter(|&(_, d2)| d2 <= pick_r2)
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .map(|(j, _)| j);
                }

                if pointer.primary_down() && (view.dragging.is_some() || pos_in_this) {
                    if let Some(j) = view.dragging {
                        view.pos[j] = view.cam.s2w(origin, pos);
                    } else if hovered {
                        view.cam.pan += drag_delta;
                    }
                } else if pointer.primary_released() {
                    view.dragging = None;
                }
            }
        }
    }

    fn render_view(view: &View, painter: &egui::Painter, origin: Pos2) {
        let z = view.cam.zoom;
        let node_r = (14.0 * z).clamp(6.0, 40.0);
        let label_dy = 22.0 * z;
        let font_px = (14.0 * z).clamp(9.0, 48.0);
        let font_id = egui::FontId::monospace(font_px);

        let dragging = view.dragging.map(|i| i as i32); // NodeIndex.index() is i32 under the hood

        // Edges: turn bright green if they touch the dragged node
        for e in view.g.edge_indices() {
            let (u, w) = view.g.edge_endpoints(e).unwrap();
            let pu = view.cam.w2s(origin, view.pos[u.index()]);
            let pw = view.cam.w2s(origin, view.pos[w.index()]);

            let edge_color = if dragging.is_some()
                && (u.index() as i32 == dragging.unwrap() || w.index() as i32 == dragging.unwrap())
            {
                Color32::from_hex("#7FEE64").unwrap()
            } else {
                Color32::DARK_GRAY
            };
            draw_arrow(painter, pu, pw, z, edge_color);
        }

        // Nodes: dragged node becomes bright green
        for nidx in view.g.node_indices() {
            let p = view.cam.w2s(origin, view.pos[nidx.index()]);
            let (label, _, _, _) = &view.g[nidx];

            let override_color = if dragging.is_some() && nidx.index() as i32 == dragging.unwrap() {
                Some(Color32::from_hex("#7FEE64").unwrap())
            } else {
                None
            };

            draw_node(
                painter,
                p,
                node_r,
                &view.g[nidx],
                view.cam.display_mode,
                override_color,
            );

            painter.text(
                p + Vec2::new(0.0, -label_dy),
                egui::Align2::CENTER_CENTER,
                label,
                font_id.clone(),
                Color32::from_hex("#7FEE64").unwrap(),
            );
        }
    }

    fn draw_central_panel(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default().show(ctx, |ui| {
            let n = self.views.len().max(1);
            ui.columns(n, |cols| {
                for i in 0..n {
                    let ui_col = &mut cols[i];

                    // Stable painter per column
                    let viewport = ui_col.available_size();
                    let (resp, painter) =
                        ui_col.allocate_painter(viewport, egui::Sense::click_and_drag());
                    let origin = resp.rect.min;

                    // Fit per view
                    let view = &mut self.views[i];
                    if view.need_fit {
                        let (min_w, max_w) = world_bounds(
                            ui_col,
                            &view.g,
                            &view.pos,
                            &egui::FontId::proportional(14.0),
                        );
                        if min_w.x.is_finite() {
                            view.cam.fit(origin, viewport, min_w, max_w, 40.0);
                        }
                        view.need_fit = false;
                    }

                    // Handle input
                    Self::handle_view_input(view, ui_col, &resp, origin);

                    // Render view
                    Self::render_view(view, &painter, origin);

                    // Draw divider between views
                    if n > 1 && i < n - 1 {
                        let rect = resp.rect;
                        let right_edge = rect.right_center();
                        painter.line_segment(
                            [
                                right_edge - egui::vec2(0.0, rect.height() / 2.0),
                                right_edge + egui::vec2(0.0, rect.height() / 2.0),
                            ],
                            egui::Stroke::new(3.0, Color32::from_rgb(34, 197, 94)),
                        );
                    }
                }
            });
        });
    }
}

impl eframe::App for Debugger {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.setup_visuals_and_background(ctx);
        self.handle_global_shortcuts(ctx);
        if self.should_close || ctx.input(|i| i.viewport().close_requested()) {
            ctx.send_viewport_cmd(egui::ViewportCommand::Close);
            return;
        }
        self.draw_title_bar(ctx);
        self.draw_top_panel(ctx);
        self.draw_central_panel(ctx);
    }

    fn clear_color(&self, _visuals: &egui::Visuals) -> [f32; 4] {
        egui::Rgba::TRANSPARENT.to_array()
    }
}

#[derive(Clone, Copy)]
pub struct Camera {
    zoom: f32, // scale
    pan: Vec2, // screen px offset
    display_mode: DisplayMode,
}

impl Camera {
    fn new() -> Self {
        Self {
            zoom: 1.0,
            pan: Vec2::ZERO,
            display_mode: DisplayMode::TermType,
        }
    }

    #[inline]
    fn w2s(&self, origin: Pos2, w: Pos2) -> Pos2 {
        origin + (self.pan + w.to_vec2() * self.zoom)
    }
    #[inline]
    fn s2w(&self, origin: Pos2, s: Pos2) -> Pos2 {
        (((s - origin) - self.pan) / self.zoom).to_pos2()
    }
    fn zoom_at(&mut self, origin: Pos2, cursor: Pos2, factor: f32) {
        let world_before = self.s2w(origin, cursor);
        let new_zoom = (self.zoom * factor).clamp(0.1, 10.0);
        self.pan = (cursor - origin) - world_before.to_vec2() * new_zoom;
        self.zoom = new_zoom;
    }
    fn fit(
        &mut self,
        _origin: Pos2,
        viewport: Vec2,
        world_min: Pos2,
        world_max: Pos2,
        margin: f32,
    ) {
        // world extents
        let size_w = (world_max - world_min).max(Vec2::splat(1.0));
        // usable viewport *in local coords* (no origin baked in)
        let usable = (viewport - Vec2::splat(2.0 * margin)).max(Vec2::splat(1.0));

        self.zoom = (usable.x / size_w.x)
            .min(usable.y / size_w.y)
            .clamp(0.1, 10.0);

        // Map world_min -> margin, then center any extra space. Stay in local coords.
        let mapped_min_local = Vec2::splat(margin);
        let mapped_size = size_w * self.zoom;
        let extra = (usable - mapped_size).max(Vec2::ZERO) * 0.5;

        // IMPORTANT: no origin here. w2s will add origin later.
        self.pan = mapped_min_local + extra - world_min.to_vec2() * self.zoom;
    }
    #[inline]
    fn toggle_mode(&mut self) {
        self.display_mode = match self.display_mode {
            DisplayMode::TermType => DisplayMode::LoopLevel,
            DisplayMode::LoopLevel => DisplayMode::TermType,
        };
    }
}

// Small helper to draw nodes
fn draw_node(
    p: &egui::Painter,
    c: Pos2,
    r: f32,
    node: &DisplayNode,
    display_mode: DisplayMode,
    override_color: Option<Color32>,
) {
    let (_, base, shape, loop_level) = node;
    let normal = if display_mode == DisplayMode::LoopLevel && loop_level.is_some() {
        rainbow_color(loop_level.unwrap())
    } else {
        *base
    };
    let color = override_color.unwrap_or(normal);
    match shape {
        NodeShape::Circle => {
            p.circle_filled(c, r, color);
        }
        NodeShape::Triangle => {
            let pts: Vec<_> = (0..3)
                .map(|i| {
                    let ang = TAU * (i as f32) / 3.0 - FRAC_PI_2; // up
                    Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin())
                })
                .collect();
            p.add(egui::epaint::Shape::convex_polygon(
                pts.clone(),
                color,
                egui::Stroke::NONE,
            ));
        }
        NodeShape::InvertedTriangle => {
            let pts: Vec<_> = (0..3)
                .map(|i| {
                    let ang = TAU * (i as f32) / 3.0 + FRAC_PI_2; // down
                    Pos2::new(c.x + r * ang.cos(), c.y + r * ang.sin())
                })
                .collect();
            p.add(egui::epaint::Shape::convex_polygon(
                pts.clone(),
                color,
                egui::Stroke::NONE,
            ));
        }
        NodeShape::Square => {
            let pts = vec![
                Pos2::new(c.x - r, c.y - r),
                Pos2::new(c.x + r, c.y - r),
                Pos2::new(c.x + r, c.y + r),
                Pos2::new(c.x - r, c.y + r),
            ];
            p.add(egui::epaint::Shape::convex_polygon(
                pts.clone(),
                color,
                egui::Stroke::NONE,
            ));
        }
    }
}

fn rainbow_color(i: usize) -> egui::Color32 {
    // Map 0..=10 across 0..=360 degrees
    let hue = (9.0 - i as f32) / 10.0 * 360.0;
    hsv_to_rgb(hue, 1.0, 1.0)
}

// Simple HSV → Color32
fn hsv_to_rgb(h: f32, s: f32, v: f32) -> egui::Color32 {
    let c = v * s;
    let x = c * (1.0 - ((h / 60.0) % 2.0 - 1.0).abs());
    let m = v - c;

    let (r, g, b) = match h as i32 {
        0..=59 => (c, x, 0.0),
        60..=119 => (x, c, 0.0),
        120..=179 => (0.0, c, x),
        180..=239 => (0.0, x, c),
        240..=299 => (x, 0.0, c),
        _ => (c, 0.0, x),
    };

    egui::Color32::from_rgb(
        ((r + m) * 255.0) as u8,
        ((g + m) * 255.0) as u8,
        ((b + m) * 255.0) as u8,
    )
}

fn world_bounds(
    ui: &mut egui::Ui,
    g: &DisplayGraph,
    pos: &[Pos2],
    font_id: &egui::FontId,
) -> (Pos2, Pos2) {
    const R: f32 = 14.0;
    const DY: f32 = 22.0;
    let mut min = Pos2::new(f32::INFINITY, f32::INFINITY);
    let mut max = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
    ui.fonts(|f| {
        for n in g.node_indices() {
            let p = pos[n.index()];
            let galley = f.layout_no_wrap(g[n].0.clone(), font_id.clone(), Color32::WHITE);
            let hw = galley.size().x * 0.5;
            let hh = galley.size().y * 0.5;
            let label_c = Pos2::new(p.x, p.y - DY);
            let lmin = Pos2::new(label_c.x - hw, label_c.y - hh);
            let lmax = Pos2::new(label_c.x + hw, label_c.y + hh);
            let nmin = Pos2::new(p.x - R, p.y - R);
            let nmax = Pos2::new(p.x + R, p.y + R);
            min.x = min.x.min(lmin.x.min(nmin.x));
            min.y = min.y.min(lmin.y.min(nmin.y));
            max.x = max.x.max(lmax.x.max(nmax.x));
            max.y = max.y.max(lmax.y.max(nmax.y));
        }
    });
    if !min.x.is_finite() {
        (Pos2::ZERO, Pos2::new(400.0, 300.0))
    } else {
        (min, max)
    }
}

/// Simple layered DAG layout
fn layered_layout(g: &DisplayGraph, dx: f32, layer_dy: f32, layer_cx: f32) -> Vec<Pos2> {
    use std::collections::VecDeque;

    let n = g.node_count();
    if n == 0 {
        return vec![];
    }

    // 1) Compute depth from sinks upward: sink=0, parent=child+1
    let mut depth = vec![usize::MAX; n];
    let mut q = VecDeque::new();
    for u in g.node_indices() {
        if g.neighbors_directed(u, Direction::Outgoing)
            .next()
            .is_none()
        {
            depth[u.index()] = 0;
            q.push_back(u);
        }
    }
    if q.is_empty() {
        if let Some(u) = g.node_indices().next() {
            depth[u.index()] = 0;
            q.push_back(u);
        }
    }
    while let Some(child) = q.pop_front() {
        let cd = depth[child.index()];
        for parent in g.neighbors_directed(child, Direction::Incoming) {
            let want = cd + 1;
            if depth[parent.index()] == usize::MAX || depth[parent.index()] < want {
                depth[parent.index()] = want;
                q.push_back(parent);
            }
        }
    }
    // Any leftover (disconnected) → one row above current top so they end up at the very top after flip
    let max_seen = depth
        .iter()
        .filter(|d| **d != usize::MAX)
        .copied()
        .max()
        .unwrap_or(0);
    for u in g.node_indices() {
        if depth[u.index()] == usize::MAX {
            depth[u.index()] = max_seen + 1;
        }
    }

    // 2) Flip so layout flows top→bottom (inputs at top, outputs at bottom)
    let max_d = *depth.iter().max().unwrap();
    let layer_of = |u: NodeIndex| max_d - depth[u.index()];

    // 3) Bucket by flipped layer (stable)
    let mut by_layer: Vec<Vec<NodeIndex>> = vec![vec![]; max_d + 1];
    for u in g.node_indices() {
        by_layer[layer_of(u)].push(u);
    }
    for layer in &mut by_layer {
        layer.sort_by_key(|u| u.index());
    }

    // 4) Place nodes
    let mut pos = vec![Pos2::ZERO; n];
    for (layer, nodes) in by_layer.into_iter().enumerate() {
        let w = (nodes.len().saturating_sub(1)) as f32 * dx;
        let y = (layer as f32) * layer_dy + 80.0;
        for (k, u) in nodes.into_iter().enumerate() {
            let x = (k as f32) * dx - w * 0.5 + layer_cx;
            pos[u.index()] = Pos2::new(x, y);
        }
    }
    pos
}

// replace your draw_arrow with a zoom-aware version
fn draw_arrow(p: &egui::Painter, from: Pos2, to: Pos2, zoom: f32, color: Color32) {
    let w = (2.0 * zoom).clamp(1.0, 4.0);
    let stroke = egui::Stroke::new(w, color);
    p.line_segment([from, to], stroke);
}

pub trait TermToString {
    fn term_to_string(&self) -> (String, Color32, NodeShape);
}

pub trait EdgeToString {
    fn edge_to_string(&self) -> String;
}

impl EdgeToString for usize {
    fn edge_to_string(&self) -> String {
        self.to_string()
    }
}

impl EdgeToString for () {
    fn edge_to_string(&self) -> String {
        "".to_string()
    }
}

impl EdgeToString for (usize, usize) {
    fn edge_to_string(&self) -> String {
        format!("{}, {}", self.0, self.1)
    }
}

impl TermToString for Term {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            match self {
                Term::App(a, _) => a.to_string(),
                Term::Lit(l) => l.to_string(),
                Term::Var(v) => v.to_string(),
            },
            Color32::GREEN,
            NodeShape::Circle,
        )
    }
}

impl TermToString for usize {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (self.to_string(), Color32::GREEN, NodeShape::Circle)
    }
}

impl TermToString for String {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (self.clone(), Color32::GREEN, NodeShape::Circle)
    }
}

impl TermToString for (Term, usize) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        let s = match &self.0 {
            Term::App(a, _) => a.to_string(),
            Term::Lit(l) => l.to_string(),
            Term::Var(v) => v.to_string(),
        };
        (
            format!("{s}[{}]", self.1),
            Color32::GREEN,
            NodeShape::Circle,
        )
    }
}

impl TermToString for Kernel {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            if self.code.starts_with("Inputs") {
                "Inputs".to_string()
            } else if self.code.starts_with("Outputs") {
                "Outputs".to_string()
            } else {
                format!(
                    "Kernel {:?} {:?} -> {:?}",
                    self.grid, self.threadblock, self.outputs
                )
            },
            Color32::GOLD,
            NodeShape::Square,
        )
    }
}

impl TermToString for GraphTerm {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        match self {
            GraphTerm::Add => (
                "Add".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Mul => (
                "Mul".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Max => (
                "Max".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Exp2 => (
                "Exp2".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Log2 => (
                "Log2".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Sin => (
                "Sin".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Recip => (
                "Recip".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Neg => (
                "Neg".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Sqrt => (
                "Sqrt".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::Mod => (
                "Mod".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::LessThan => (
                "LessThan".to_string(),
                Color32::from_hex("#55a042").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::TCMatmul {
                a_k_stride,
                b_k_stride,
                a_inner_stride,
                b_inner_stride,
                c_inner_stride,
                k_outer_loops,
            } => (
                format!(
                    "TCMatmul ({a_k_stride}, {b_k_stride}, {a_inner_stride}, {b_inner_stride}, {c_inner_stride}, {k_outer_loops})"
                ),
                Color32::from_hex("#bf1919").unwrap(),
                NodeShape::Circle,
            ),
            GraphTerm::LoopIn {
                range,
                stride,
                marker,
            } => (
                format!("LoopIn ({range}; {stride}; ({marker}))"),
                Color32::from_hex("#40b2e9").unwrap(),
                NodeShape::InvertedTriangle,
            ),
            GraphTerm::LoopOut {
                range,
                stride,
                marker,
            } => (
                format!("LoopOut ({range}; {stride}; ({marker}))"),
                Color32::from_hex("#4340e9").unwrap(),
                NodeShape::Triangle,
            ),
            GraphTerm::GMEM { label } => (
                format!("GMEM ({label})"),
                Color32::from_hex("#a6ad1d").unwrap(),
                NodeShape::Square,
            ),
            GraphTerm::SMEM => ("SMEM".to_string(), Color32::YELLOW, NodeShape::Square),
            GraphTerm::Custom(_) => ("CustomKernel".to_string(), Color32::GRAY, NodeShape::Circle),
            GraphTerm::Diff(d) => (format!("Diff({d})"), Color32::GRAY, NodeShape::Circle),
            GraphTerm::SMEMLoad => ("SMEMLoad".to_string(), Color32::YELLOW, NodeShape::Circle),
            GraphTerm::SMEMRead => ("SMEMRead".to_string(), Color32::YELLOW, NodeShape::Circle),
            GraphTerm::Break => ("Break".to_string(), Color32::WHITE, NodeShape::Circle),
        }
    }
}

impl TermToString for (GraphTerm, usize) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!("{} [{}]", self.0.term_to_string().0, self.1),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, Vec<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} {:?} {{{:?}}}",
                self.0.term_to_string().0,
                self.1,
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<Expression>, FxHashSet<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} {:?} {{{:?}}}",
                self.0.term_to_string().0,
                self.1,
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

impl TermToString for (GraphTerm, Vec<String>, Vec<usize>) {
    fn term_to_string(&self) -> (String, Color32, NodeShape) {
        (
            format!(
                "{} [{}] {{{:?}}}",
                self.0.term_to_string().0,
                self.1.len(),
                self.2
            ),
            self.0.term_to_string().1,
            NodeShape::Circle,
        )
    }
}

/// View a debug graph in the browser
pub fn display_graph2(
    graph: &StableGraph<impl TermToString, impl EdgeToString, Directed, u32>,
    mark_nodes: &[(NodeIndex, &str)],
) {
    let mut new_graph = StableGraph::new();
    let mut map = std::collections::HashMap::new();
    for node in graph.node_indices() {
        map.insert(
            node,
            new_graph.add_node(graph.node_weight(node).unwrap().term_to_string().0),
        );
    }
    for edge in graph.edge_indices() {
        let weight = graph.edge_weight(edge).unwrap();
        let (src, dest) = graph.edge_endpoints(edge).unwrap();
        new_graph.add_edge(map[&src], map[&dest], weight.edge_to_string());
    }
    let mut graph_string = luminal::prelude::petgraph::dot::Dot::with_config(
        &new_graph,
        &[luminal::prelude::petgraph::dot::Config::EdgeIndexLabel],
    )
    .to_string();
    let re = regex::Regex::new(r#"label\s*=\s*"\d+""#).unwrap();
    graph_string = re.replace_all(&graph_string, "").to_string();
    for (n, color) in mark_nodes {
        graph_string = graph_string.replace(
            &format!("    {} [ label =", n.index()),
            &format!(
                "    {} [ style=\"filled\" fillcolor=\"{color}\" label =",
                n.index(),
            ),
        );
    }

    let url = format!(
        "https://dreampuf.github.io/GraphvizOnline/#{}",
        urlencoding::encode(&graph_string)
    );
    if let Err(e) = webbrowser::open(&url) {
        panic!("Error displaying graph: {:?}", e);
    }
}
