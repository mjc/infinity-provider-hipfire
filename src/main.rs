//! Standalone binary serving the Hipfire [`ModelProvider`] over a Unix
//! socket (see `infinity_provider_protocol::remote`).
//!
//! Prints the generated socket path as the first stdout line so a
//! supervisor (for example `infinity-daemon`) can discover it. Logs go to stderr.

use std::io::Write;
use std::sync::Arc;

use infinity_provider_hipfire::HipfireProvider;
use infinity_provider_protocol::remote::serve_provider;
use tracing_subscriber::EnvFilter;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

#[tokio::main]
async fn main() -> Result<(), BoxError> {
    // stdout is reserved for the socket path; log to stderr.
    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env())
        .with_writer(std::io::stderr)
        .with_ansi(false)
        .init();

    let provider = Arc::new(HipfireProvider::from_env());
    let (socket_path, server) = serve_provider(provider)?;

    println!("{}", socket_path.display());
    std::io::stdout().flush()?;

    server.await;
    Ok(())
}
