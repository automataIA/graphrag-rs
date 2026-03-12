use indicatif::{ProgressBar, ProgressStyle};

fn main() {
    let style = ProgressStyle::default_bar()
        .template("   [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} chunks ({eta})")
        .unwrap()
        .progress_chars("=>-");

    let pb = ProgressBar::new(100).with_style(style);
    pb.finish_with_message("Done");
}
