use axum::{
    Router,
    body::Bytes,
    http::header,
    response::{Html, IntoResponse, Redirect},
    routing::get,
};

async fn index() -> Html<String> {
    Html(include_str!("../static/index.html").to_string())
}

async fn intro_quiz() -> Html<String> {
    Html(include_str!("../static/intro_quiz.html").to_string())
}

async fn redirect() -> Redirect {
    Redirect::to("https://luminalai.com")
}

async fn favicon() -> impl IntoResponse {
    // Embed at compile time
    let ico = Bytes::from_static(include_bytes!("../../images/favicon.ico"));
    (
        [
            (header::CONTENT_TYPE, "image/x-icon"),
            (header::CACHE_CONTROL, "public, max-age=31536000, immutable"),
        ],
        ico,
    )
}

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(index))
        .route("/intro_quiz", get(intro_quiz))
        .route("/docs/introduction", get(redirect))
        .route_service("/favicon.ico", get(favicon));

    println!("Running on port 3000...");
    let listener = tokio::net::TcpListener::bind("0.0.0.0:3000").await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
