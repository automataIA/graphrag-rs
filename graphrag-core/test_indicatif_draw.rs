use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};

fn main() {
    // We want to see if creating it prints anything before setting style.
    let pb = ProgressBar::with_draw_target(100, ProgressDrawTarget::stderr());
    // Sleep to see if it drew
    std::thread::sleep(std::time::Duration::from_millis(100));
    let style = ProgressStyle::default_bar()
        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
        .unwrap()
        .progress_chars("=>-");
    pb.set_style(style);
    pb.finish_with_message("Done");
}
