use crate::plotly_support::use_plotly_chart;
use interpn::{
    GridInterpMethod, GridKind, Left1D, Linear1D, LinearHoldLast1D, Nearest1D, RectilinearGrid1D,
    RegularGrid1D, Right1D, interpn_serial, multibspline::regular as bspline_regular,
    one_dim::Interp1D,
};
use leptos::prelude::*;
use plotly::{
    Layout, Plot, Scatter,
    common::{DashType, Line, Marker, Mode, Title},
    layout::Axis,
};

const SAMPLE_COUNT: usize = 500;
const BSPLINE_KNOTS: usize = 8;
const DEIMOS_PRIMARY: &str = "#8232ba";
const DEIMOS_ACCENT: &str = "#ac37ff";
const DEIMOS_WARNING: &str = "#ffd166";
const DEIMOS_DANGER: &str = "#f26d7d";
const REFERENCE_LINE: &str = "#000000";

#[component]
pub fn App() -> impl IntoView {
    let (light_mode, set_light_mode) = signal(false);
    let (truth_function, set_truth_function) = signal(TruthFunction::SineMix);
    let (grid_kind, set_grid_kind) = signal(GridChoice::Regular);
    let (method, set_method) = signal(MethodChoice::Linear);
    let (grid_points, set_grid_points) = signal(10_usize);
    let (show_error, set_show_error) = signal(false);
    let (bspline_input_mode, set_bspline_input_mode) = signal(BsplineInputMode::NodalValues);
    let (knot_values, set_knot_values) =
        signal(vec![0.0, 0.55, 0.95, 0.35, -0.45, -0.9, -0.35, 0.25]);

    let comparison_inputs = move || ComparisonInputs {
        truth_function: truth_function.get(),
        grid_kind: grid_kind.get(),
        method: method.get(),
        grid_points: grid_points.get(),
        show_error: show_error.get(),
    };
    let comparison = Memo::new(move |_| build_comparison(comparison_inputs()));
    let spline =
        Memo::new(move |_| build_bspline_demo(bspline_input_mode.get(), knot_values.get()));

    use_plotly_chart("comparison-plot", move || {
        build_comparison_plot(comparison.get())
    });
    use_plotly_chart("bspline-plot", move || build_bspline_plot(spline.get()));

    let comparison_status = move || comparison.get().status;
    let spline_status = move || spline.get().status;

    view! {
        <div class=move || {
            if light_mode.get() {
                "app-shell light-mode"
            } else {
                "app-shell"
            }
        }>
            <aside class="sidebar">
                <h1 class="sidebar-title">"interpn"</h1>
                <button
                    class=move || {
                        if light_mode.get() {
                            "theme-toggle active"
                        } else {
                            "theme-toggle"
                        }
                    }
                    attr:aria-pressed=move || light_mode.get().to_string()
                    on:click=move |_| set_light_mode.update(|enabled| *enabled = !*enabled)
                >
                    <span class="theme-toggle-icon" aria-hidden="true"></span>
                    <span>"Light mode"</span>
                </button>
                <nav>
                    <a href="#comparison">"1D Comparison"</a>
                    <a href="#bspline">"B-Spline Coefficients"</a>
                </nav>
            </aside>

            <main class="content">
                <section id="comparison" class="section-block">
                    <header class="section-header">
                        <p class="eyebrow">"1D"</p>
                        <h2>"Interpolation comparison"</h2>
                    </header>

                    <div class="control-layout">
                        <aside class="control-card">
                            <div class="control-row">
                                <label for="truth-function">"Truth function"</label>
                                <select
                                    id="truth-function"
                                    prop:value=move || truth_function.get().as_key().to_string()
                                    on:change=move |ev| {
                                        set_truth_function
                                            .set(TruthFunction::from_form_value(&event_target_value(&ev)));
                                    }
                                >
                                    <option value="sine_mix">"Sine mix"</option>
                                    <option value="runge">"Runge"</option>
                                    <option value="corner">"Corner"</option>
                                    <option value="chirp">"Chirp"</option>
                                </select>
                            </div>

                            <div class="control-row">
                                <label for="grid-kind">"Grid"</label>
                                <select
                                    id="grid-kind"
                                    prop:value=move || grid_kind.get().as_key().to_string()
                                    on:change=move |ev| {
                                        set_grid_kind
                                            .set(GridChoice::from_form_value(&event_target_value(&ev)));
                                    }
                                >
                                    <option value="regular">"Regular"</option>
                                    <option value="rectilinear">"Rectilinear"</option>
                                </select>
                            </div>

                            <div class="control-row">
                                <label for="method">"Method"</label>
                                <select
                                    id="method"
                                    prop:value=move || method.get().as_key().to_string()
                                    on:change=move |ev| {
                                        set_method
                                            .set(MethodChoice::from_form_value(&event_target_value(&ev)));
                                    }
                                >
                                    <option value="linear">"Linear"</option>
                                    <option value="linear_hold_last">"Linear hold-last"</option>
                                    <option value="cubic">"Cubic Hermite"</option>
                                    <option value="bspline">"Cubic B-spline"</option>
                                    <option value="nearest">"Nearest"</option>
                                    <option value="left">"Left hold"</option>
                                    <option value="right">"Right hold"</option>
                                </select>
                            </div>

                            <div class="control-row">
                                <label for="grid-points">"Grid points"</label>
                                <output>{move || grid_points.get().to_string()}</output>
                                <input
                                    id="grid-points"
                                    type="range"
                                    min="4"
                                    max="28"
                                    step="1"
                                    prop:value=move || grid_points.get().to_string()
                                    on:input=move |ev| {
                                        if let Ok(value) = event_target_value(&ev).parse::<usize>() {
                                            set_grid_points.set(value.clamp(4, 28));
                                        }
                                    }
                                />
                            </div>

                            <div class="control-row checkbox-row">
                                <label for="show-error">"Show absolute error"</label>
                                <input
                                    id="show-error"
                                    type="checkbox"
                                    prop:checked=move || show_error.get()
                                    on:change=move |ev| set_show_error.set(event_target_checked(&ev))
                                />
                            </div>

                            <p class="status-line">{comparison_status}</p>
                        </aside>

                        <article class="plot-card">
                            <div id="comparison-plot" class="plot-surface"></div>
                        </article>
                    </div>
                </section>

                <section id="bspline" class="section-block">
                    <header class="section-header">
                        <p class="eyebrow">"B-Spline"</p>
                        <h2>"Coefficient panel"</h2>
                    </header>

                    <div class="control-layout">
                        <aside class="control-card">
                            <div class="control-row">
                                <label for="bspline-input-mode">"Slider interpretation"</label>
                                <select
                                    id="bspline-input-mode"
                                    prop:value=move || bspline_input_mode.get().as_key().to_string()
                                    on:change=move |ev| {
                                        set_bspline_input_mode
                                            .set(BsplineInputMode::from_form_value(&event_target_value(&ev)));
                                    }
                                >
                                    <option value="nodal_values">"Nodal data values"</option>
                                    <option value="coefficients">"B-spline coefficients"</option>
                                </select>
                            </div>
                            {(0..BSPLINE_KNOTS)
                                .map(|index| {
                                    view! {
                                        <div class="control-row">
                                            <output>
                                                {move || format!("{:.2}", knot_values.get()[index])}
                                            </output>
                                            <input
                                                id=format!("knot-{index}")
                                                attr:aria-label=move || {
                                                    format!(
                                                        "{} {}",
                                                        bspline_input_mode.get().slider_label(),
                                                        index + 1,
                                                    )
                                                }
                                                type="range"
                                                min="-1.50"
                                                max="1.50"
                                                step="0.05"
                                                prop:value=move || knot_values.get()[index].to_string()
                                                on:input=move |ev| {
                                                    if let Ok(value) = event_target_value(&ev).parse::<f64>() {
                                                        set_knot_values
                                                            .update(|values| values[index] = value.clamp(-1.5, 1.5));
                                                    }
                                                }
                                            />
                                        </div>
                                    }
                                })
                                .collect_view()}
                            <p class="status-line">{spline_status}</p>
                        </aside>

                        <article class="plot-card">
                            <div id="bspline-plot" class="plot-surface"></div>
                        </article>
                    </div>
                </section>
            </main>
        </div>
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BsplineInputMode {
    NodalValues,
    Coefficients,
}

impl BsplineInputMode {
    const fn as_key(self) -> &'static str {
        match self {
            Self::NodalValues => "nodal_values",
            Self::Coefficients => "coefficients",
        }
    }

    const fn slider_label(self) -> &'static str {
        match self {
            Self::NodalValues => "Data",
            Self::Coefficients => "Coeff",
        }
    }

    const fn marker_label(self) -> &'static str {
        match self {
            Self::NodalValues => "nodal data",
            Self::Coefficients => "B-spline coefficients",
        }
    }

    const fn plot_title(self) -> &'static str {
        match self {
            Self::NodalValues => "Cubic B-spline solved from nodal data",
            Self::Coefficients => "Cubic B-spline from direct coefficients",
        }
    }

    fn from_form_value(value: &str) -> Self {
        match value {
            "coefficients" => Self::Coefficients,
            _ => Self::NodalValues,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum TruthFunction {
    SineMix,
    Runge,
    Corner,
    Chirp,
}

impl TruthFunction {
    const fn as_key(self) -> &'static str {
        match self {
            Self::SineMix => "sine_mix",
            Self::Runge => "runge",
            Self::Corner => "corner",
            Self::Chirp => "chirp",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::SineMix => "Sine mix",
            Self::Runge => "Runge",
            Self::Corner => "Corner",
            Self::Chirp => "Chirp",
        }
    }

    fn from_form_value(value: &str) -> Self {
        match value {
            "runge" => Self::Runge,
            "corner" => Self::Corner,
            "chirp" => Self::Chirp,
            _ => Self::SineMix,
        }
    }

    fn eval(self, x: f64) -> f64 {
        match self {
            Self::SineMix => {
                (std::f64::consts::TAU * x).sin() + 0.28 * (5.0 * std::f64::consts::PI * x).cos()
            }
            Self::Runge => 1.0 / (1.0 + 18.0 * x * x),
            Self::Corner => {
                if x < -0.2 {
                    -0.55 + 0.25 * (4.0 * x).sin()
                } else if x < 0.42 {
                    0.25 + 1.1 * x
                } else {
                    1.05 - 0.8 * x
                }
            }
            Self::Chirp => (8.0 * x * x + 1.5 * x).sin() * (-0.7 * (x + 1.0)).exp(),
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum GridChoice {
    Regular,
    Rectilinear,
}

impl GridChoice {
    const fn as_key(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Rectilinear => "rectilinear",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Regular => "regular",
            Self::Rectilinear => "rectilinear",
        }
    }

    const fn grid_kind(self) -> GridKind {
        match self {
            Self::Regular => GridKind::Regular,
            Self::Rectilinear => GridKind::Rectilinear,
        }
    }

    fn from_form_value(value: &str) -> Self {
        match value {
            "rectilinear" => Self::Rectilinear,
            _ => Self::Regular,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum MethodChoice {
    Linear,
    LinearHoldLast,
    Cubic,
    Bspline,
    Nearest,
    Left,
    Right,
}

impl MethodChoice {
    const fn as_key(self) -> &'static str {
        match self {
            Self::Linear => "linear",
            Self::LinearHoldLast => "linear_hold_last",
            Self::Cubic => "cubic",
            Self::Bspline => "bspline",
            Self::Nearest => "nearest",
            Self::Left => "left",
            Self::Right => "right",
        }
    }

    const fn label(self) -> &'static str {
        match self {
            Self::Linear => "Linear",
            Self::LinearHoldLast => "Linear hold-last",
            Self::Cubic => "Cubic Hermite",
            Self::Bspline => "Cubic B-spline",
            Self::Nearest => "Nearest",
            Self::Left => "Left hold",
            Self::Right => "Right hold",
        }
    }

    const fn grid_method(self) -> Option<GridInterpMethod> {
        match self {
            Self::Linear => Some(GridInterpMethod::Linear),
            Self::Cubic => Some(GridInterpMethod::Cubic),
            Self::Bspline => Some(GridInterpMethod::Bspline),
            Self::Nearest => Some(GridInterpMethod::Nearest),
            Self::LinearHoldLast | Self::Left | Self::Right => None,
        }
    }

    fn from_form_value(value: &str) -> Self {
        match value {
            "linear_hold_last" => Self::LinearHoldLast,
            "cubic" => Self::Cubic,
            "bspline" => Self::Bspline,
            "nearest" => Self::Nearest,
            "left" => Self::Left,
            "right" => Self::Right,
            _ => Self::Linear,
        }
    }
}

#[derive(Clone, Copy)]
struct ComparisonInputs {
    truth_function: TruthFunction,
    grid_kind: GridChoice,
    method: MethodChoice,
    grid_points: usize,
    show_error: bool,
}

#[derive(Clone, PartialEq)]
struct ComparisonData {
    title: String,
    x_eval: Vec<f64>,
    y_truth: Vec<f64>,
    y_interp: Vec<f64>,
    y_error: Vec<f64>,
    x_grid: Vec<f64>,
    y_grid: Vec<f64>,
    show_error: bool,
    status: String,
}

#[derive(Clone, PartialEq)]
struct BsplineData {
    x_eval: Vec<f64>,
    y_curve: Vec<f64>,
    x_nodes: Vec<f64>,
    y_inputs: Vec<f64>,
    y_curve_at_nodes: Vec<f64>,
    input_label: &'static str,
    title: &'static str,
    status: String,
}

fn build_comparison(inputs: ComparisonInputs) -> ComparisonData {
    let x_eval = linspace(-1.15, 1.15, SAMPLE_COUNT);
    let x_grid = build_grid(inputs.grid_kind, inputs.grid_points);
    let y_grid = x_grid
        .iter()
        .map(|&x| inputs.truth_function.eval(x))
        .collect::<Vec<_>>();
    let y_truth = x_eval
        .iter()
        .map(|&x| inputs.truth_function.eval(x))
        .collect::<Vec<_>>();

    let (y_interp, status) =
        interpolate_curve(inputs.grid_kind, inputs.method, &x_grid, &y_grid, &x_eval)
            .map(|values| {
                let rms = rms_error(&values, &y_truth);
                (values, format!("RMS error: {rms:.3e}"))
            })
            .unwrap_or_else(|msg| (vec![f64::NAN; x_eval.len()], format!("Error: {msg}")));
    let y_error = y_interp
        .iter()
        .zip(y_truth.iter())
        .map(|(interp, truth)| (interp - truth).abs())
        .collect::<Vec<_>>();

    ComparisonData {
        title: format!(
            "{} on a {} grid with {}",
            inputs.truth_function.label(),
            inputs.grid_kind.label(),
            inputs.method.label()
        ),
        x_eval,
        y_truth,
        y_interp,
        y_error,
        x_grid,
        y_grid,
        show_error: inputs.show_error,
        status,
    }
}

fn interpolate_curve(
    grid_kind: GridChoice,
    method: MethodChoice,
    x_grid: &[f64],
    y_grid: &[f64],
    x_eval: &[f64],
) -> Result<Vec<f64>, &'static str> {
    match method.grid_method() {
        Some(grid_method) => {
            let grids = &[x_grid];
            let obs = &[x_eval];
            let mut out = vec![0.0; x_eval.len()];
            interpn_serial(
                grids,
                y_grid,
                obs,
                &mut out,
                grid_method,
                Some(grid_kind.grid_kind()),
                true,
                None,
            )?;
            Ok(out)
        }
        None => interpolate_one_dim(grid_kind, method, x_grid, y_grid, x_eval),
    }
}

fn interpolate_one_dim(
    grid_kind: GridChoice,
    method: MethodChoice,
    x_grid: &[f64],
    y_grid: &[f64],
    x_eval: &[f64],
) -> Result<Vec<f64>, &'static str> {
    let mut out = vec![0.0; x_eval.len()];
    match grid_kind {
        GridChoice::Regular => {
            let grid = RegularGrid1D::new(x_grid[0], x_grid[1] - x_grid[0], y_grid)?;
            match method {
                MethodChoice::LinearHoldLast => {
                    LinearHoldLast1D::new(grid).eval(x_eval, &mut out)?
                }
                MethodChoice::Left => Left1D::new(grid).eval(x_eval, &mut out)?,
                MethodChoice::Right => Right1D::new(grid).eval(x_eval, &mut out)?,
                _ => Linear1D::new(grid).eval(x_eval, &mut out)?,
            }
        }
        GridChoice::Rectilinear => {
            let grid = RectilinearGrid1D::new(x_grid, y_grid)?;
            match method {
                MethodChoice::LinearHoldLast => {
                    LinearHoldLast1D::new(grid).eval(x_eval, &mut out)?
                }
                MethodChoice::Left => Left1D::new(grid).eval(x_eval, &mut out)?,
                MethodChoice::Right => Right1D::new(grid).eval(x_eval, &mut out)?,
                MethodChoice::Nearest => Nearest1D::new(grid).eval(x_eval, &mut out)?,
                _ => Linear1D::new(grid).eval(x_eval, &mut out)?,
            }
        }
    }
    Ok(out)
}

fn build_bspline_demo(input_mode: BsplineInputMode, y_inputs: Vec<f64>) -> BsplineData {
    let x_nodes = linspace(-1.0, 1.0, y_inputs.len());
    let x_eval = linspace(-1.1, 1.1, SAMPLE_COUNT);
    let mut y_curve = vec![0.0; x_eval.len()];
    let mut y_curve_at_nodes = vec![0.0; x_nodes.len()];
    let eval_result = eval_bspline(input_mode, &x_nodes, &y_inputs, &x_eval, &mut y_curve)
        .and_then(|()| {
            eval_bspline(
                input_mode,
                &x_nodes,
                &y_inputs,
                &x_nodes,
                &mut y_curve_at_nodes,
            )
        });
    let status = eval_result
        .map(|()| match input_mode {
            BsplineInputMode::NodalValues => {
                "Solved coefficients from nodal data; curve passes through the data values."
                    .to_string()
            }
            BsplineInputMode::Coefficients => {
                "Passed coefficients directly; curve values at nodes are plotted separately."
                    .to_string()
            }
        })
        .unwrap_or_else(|msg| {
            y_curve.fill(f64::NAN);
            y_curve_at_nodes.fill(f64::NAN);
            format!("Error: {msg}")
        });

    BsplineData {
        x_eval,
        y_curve,
        x_nodes,
        y_inputs,
        y_curve_at_nodes,
        input_label: input_mode.marker_label(),
        title: input_mode.plot_title(),
        status,
    }
}

fn eval_bspline(
    input_mode: BsplineInputMode,
    x_nodes: &[f64],
    y_inputs: &[f64],
    x_eval: &[f64],
    out: &mut [f64],
) -> Result<(), &'static str> {
    match input_mode {
        BsplineInputMode::NodalValues => interpn_serial(
            &[x_nodes],
            y_inputs,
            &[x_eval],
            out,
            GridInterpMethod::Bspline,
            Some(GridKind::Regular),
            true,
            None,
        ),
        BsplineInputMode::Coefficients => bspline_regular::interpn(
            &[x_nodes.len()],
            &[x_nodes[0]],
            &[x_nodes[1] - x_nodes[0]],
            y_inputs,
            true,
            &[x_eval],
            out,
        ),
    }
}

fn build_comparison_plot(data: ComparisonData) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(data.x_eval.clone(), data.y_truth)
            .mode(Mode::Lines)
            .name("truth")
            .line(Line::new().color(REFERENCE_LINE).width(3.0)),
    );
    plot.add_trace(
        Scatter::new(data.x_eval.clone(), data.y_interp)
            .mode(Mode::Lines)
            .name("interpolated")
            .line(Line::new().color(DEIMOS_PRIMARY).width(3.0)),
    );
    plot.add_trace(
        Scatter::new(data.x_grid, data.y_grid)
            .mode(Mode::Markers)
            .name("grid samples")
            .marker(Marker::new().color(DEIMOS_DANGER).size(10)),
    );
    if data.show_error {
        plot.add_trace(
            Scatter::new(data.x_eval, data.y_error)
                .mode(Mode::Lines)
                .name("absolute error")
                .line(
                    Line::new()
                        .color(DEIMOS_WARNING)
                        .width(2.0)
                        .dash(DashType::Dash),
                ),
        );
    }
    plot.set_layout(line_layout(&data.title, "x", "y"));
    plot
}

fn build_bspline_plot(data: BsplineData) -> Plot {
    let mut plot = Plot::new();
    plot.add_trace(
        Scatter::new(data.x_eval, data.y_curve)
            .mode(Mode::Lines)
            .name("B-spline curve")
            .line(Line::new().color(DEIMOS_PRIMARY).width(3.0)),
    );
    plot.add_trace(
        Scatter::new(data.x_nodes.clone(), data.y_inputs)
            .mode(Mode::Markers)
            .name(data.input_label)
            .marker(Marker::new().color(DEIMOS_DANGER).size(12)),
    );
    plot.add_trace(
        Scatter::new(data.x_nodes, data.y_curve_at_nodes)
            .mode(Mode::Markers)
            .name("curve at nodes")
            .marker(Marker::new().color(DEIMOS_ACCENT).size(8)),
    );
    plot.set_layout(line_layout(data.title, "x", "value"));
    plot
}

fn line_layout(title: &str, x_title: &str, y_title: &str) -> Layout {
    Layout::new()
        .title(Title::with_text(title))
        .x_axis(Axis::new().title(Title::with_text(x_title)))
        .y_axis(Axis::new().title(Title::with_text(y_title)))
}

fn build_grid(kind: GridChoice, n: usize) -> Vec<f64> {
    match kind {
        GridChoice::Regular => linspace(-1.0, 1.0, n),
        GridChoice::Rectilinear => (0..n)
            .map(|i| {
                let t = i as f64 / (n - 1) as f64;
                -1.0 + 2.0 * t.powf(1.35)
            })
            .collect(),
    }
}

fn linspace(start: f64, stop: f64, n: usize) -> Vec<f64> {
    match n {
        0 => Vec::new(),
        1 => vec![start],
        _ => (0..n)
            .map(|i| start + (stop - start) * i as f64 / (n - 1) as f64)
            .collect(),
    }
}

fn rms_error(values: &[f64], reference: &[f64]) -> f64 {
    let mse = values
        .iter()
        .zip(reference.iter())
        .map(|(value, reference)| {
            let err = value - reference;
            err * err
        })
        .sum::<f64>()
        / values.len().max(1) as f64;
    mse.sqrt()
}
