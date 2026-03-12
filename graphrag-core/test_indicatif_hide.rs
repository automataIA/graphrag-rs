use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use std::thread::sleep;
use std::time::Duration;

fn main() {
    let style = ProgressStyle::default_bar()
        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})")
        .unwrap();

    // Create hidden, then style, then show
    let pb = ProgressBar::with_draw_target(100, ProgressDrawTarget::hidden()).with_style(style);

    // Simulate some work before we reveal it
    sleep(Duration::from_millis(50));

    // Now reveal it
    pb.set_draw_target(ProgressDrawTarget::stderr());

    // Tick it so it draws
    pb.tick();

    pb.finish_with_message("Done");
}
