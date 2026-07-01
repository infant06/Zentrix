use clap::Subcommand;

/// Vector storage and search management
#[derive(Subcommand, Clone)]
pub enum VectorCommand {
    /// Show statistics for the vector storage
    Stats,

    /// Add a vector to storage
    Add {
        /// ID of the vector
        id: String,

        /// Vector elements (comma separated, e.g. "0.1,0.2,0.3")
        #[arg(long)]
        embedding: String,

        /// Optional raw string metadata
        #[arg(long)]
        metadata: Option<String>,

        /// Optional metadata (JSON string)
        #[arg(long)]
        metadata_json: Option<String>,
        
        /// Optional simple text metadata
        #[arg(long)]
        text: Option<String>,
    },

    /// Search for similar vectors
    Search {
        /// Vector elements (comma separated, e.g. "0.1,0.2,0.3")
        #[arg(long)]
        embedding: String,

        /// Number of results to return
        #[arg(long, default_value_t = 10)]
        top_k: usize,
    },
}
