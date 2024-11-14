#![allow(unused_braces)]
use actix_web::{post, web, App, HttpResponse, HttpServer, Responder};
use serde::Deserialize;
use std::fs;
use std::sync::Arc;

mod classification;

wit_bindgen::generate!({
    path: "../../wit",
    world: "ml",
});

// Struct to receive the URL in JSON format
#[derive(Deserialize)]
struct ImageRequest {
    url: String,
}

#[post("/classify")]
async fn classify_image(
    data: web::Data<AppState>,
    req: web::Json<ImageRequest>,
) -> impl Responder {
    match download_image(&req.url).await {
        Ok(image_path) => {
            let result = data.classifier.classify_image(&image_path);
            let _ = fs::remove_file(&image_path); // Clean up the downloaded image

            match result {
                Ok(top_classes) => HttpResponse::Ok().json(top_classes),
                Err(e) => HttpResponse::InternalServerError().body(format!("Classification error: {}", e)),
            }
        }
        Err(e) => HttpResponse::BadRequest().body(format!("Image download error: {}", e)),
    }
}

async fn download_image(url: &str) -> Result<String, Box<dyn std::error::Error>> {
    let response = reqwest::get(url).await?;
    if response.status().is_success() {
        let bytes = response.bytes().await?;
        let image_path = "downloaded_image.jpg";
        fs::write(&image_path, &bytes)?;
        Ok(image_path.to_string())
    } else {
        Err(format!("Failed to download image: {}", response.status()).into())
    }
}

struct AppState {
    classifier: Arc<classification::Classifier>,
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let classifier = classification::Classifier::new(
        "fixture/models/squeezenet1.1-7.onnx",
        "fixture/labels/squeezenet1.1-7.txt",
    )
    .expect("Failed to initialize classifier");

    let app_state = web::Data::new(AppState {
        classifier: Arc::new(classifier),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(app_state.clone())
            .service(classify_image)
    })
    .bind(("127.0.0.1", 8080))?
    .run()
    .await
}
