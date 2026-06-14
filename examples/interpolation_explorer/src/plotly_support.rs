use plotly::Plot;

pub fn use_plotly_chart<P>(id: &'static str, plot: P)
where
    P: Fn() -> Plot + 'static,
{
    #[cfg(target_arch = "wasm32")]
    {
        use leptos::prelude::*;

        Effect::new(move |_| {
            let mut next_plot = plot();
            let responsive_config = next_plot
                .configuration()
                .clone()
                .responsive(true)
                .autosizable(true)
                .fill_frame(false);
            next_plot.set_configuration(responsive_config);
            wasm_bindgen_futures::spawn_local(async move {
                plotly::bindings::new_plot(id, &next_plot).await;
            });
        });
    }

    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = (id, plot);
    }
}
