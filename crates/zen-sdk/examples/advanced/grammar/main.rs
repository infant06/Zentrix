//! Constrained generation using a GBNF grammar.
//!
//! Run with: `cargo run --release --example grammar -p zentrix`

use anyhow::Result;
use zen_sdk::{
    IsqBits, ModelBuilder, PagedAttentionMetaBuilder, RequestBuilder, TextMessageRole,
};

#[tokio::main]
async fn main() -> Result<()> {
    let model = ModelBuilder::new("google/gemma-4-E4B-it")
        .with_auto_isq(IsqBits::Four)
        .with_logging()
        .with_paged_attn(PagedAttentionMetaBuilder::default().build()?)
        .build()
        .await?;

    // Bullet list regex
    let request = RequestBuilder::new()
        .set_constraint(zen_sdk::Constraint::Regex(
            "(- [^\n]*\n)+(- [^\n]*)(\n\n)?".to_string(),
        ))
        .add_message(TextMessageRole::User, "Please write a few jokes.");

    let response = model.send_chat_request(request).await?;

    println!("{}", response.choices[0].message.content.as_ref().unwrap());

    Ok(())
}
