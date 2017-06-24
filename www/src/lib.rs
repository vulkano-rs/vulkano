// Copyright (c) 2017 The vulkano developers
// Licensed under the Apache License, Version 2.0
// <LICENSE-APACHE or
// http://www.apache.org/licenses/LICENSE-2.0> or the MIT
// license <LICENSE-MIT or http://opensource.org/licenses/MIT>,
// at your option. All files in the project carrying such
// notice may not be copied, modified, or distributed except
// according to those terms.

#[macro_use]
extern crate lazy_static;
extern crate markdown;
extern crate mustache;
#[macro_use]
extern crate rouille;

use std::collections::HashMap;
use std::io;
use std::net::ToSocketAddrs;
use std::sync::Mutex;
use rouille::Response;

pub fn start<A>(addr: A)
where
    A: ToSocketAddrs,
{
    rouille::start_server(addr, move |request| {
        rouille::content_encoding::apply(
            &request,
            rouille::log(request, io::stdout(), || {
                {
                    let mut r = rouille::match_assets(request, "./static");
                    if r.is_success() {
                        r.headers.push((
                            "Cache-Control".into(),
                            format!("max-age={}", 2 * 60 * 60).into(),
                        ));
                        return r;
                    }
                }

                router!(request,
                    (GET) (/) => {
                        render_template(include_str!("../content/home.md"))
                    },
                    _ => {
                        if let Some(request) = request.remove_prefix("/guides") {
                            if request.raw_url().starts_with("/01") {
                                render_template(include_str!("../content/01-getting-started.md"))
                            } else if request.raw_url().starts_with("/02") {
                                render_template(include_str!("../content/02-first-operation.md"))
                            } else if request.raw_url().starts_with("/03") {
                                render_template(include_str!("../content/03-window-swapchain.md"))
                            } else if request.raw_url().starts_with("/04") {
                                render_template(include_str!("../content/04-render-pass.md"))
                            } else if request.raw_url().starts_with("/05") {
                                render_template(include_str!("../content/05-first-triangle.md"))
                            } else {
                                Response::empty_404()
                            }
                        } else {
                            Response::empty_404()
                        }
                    }
                )
            }),
        )
    });
}

fn render_template(markdown: &'static str) -> Response {
    lazy_static! {
        static ref MAIN_TEMPLATE: mustache::Template = {
            mustache::compile_str(&include_str!("../content/template.html")).unwrap()
        };

        static ref CACHE: Mutex<HashMap<&'static str, String>> = Mutex::new(HashMap::new());
    }

    // The markdown crate somehow takes a lot of time (10s in debug, 800ms in release), so we cache
    // the compilation result.
    let mut compil_cache = CACHE.lock().unwrap();
    let html = compil_cache.entry(markdown).or_insert_with(|| {
        let body = markdown::to_html(markdown);

        let data = mustache::MapBuilder::new()
            .insert_str("body", body)
            .build();

        let mut out = Vec::new();
        MAIN_TEMPLATE.render_data(&mut out, &data).unwrap();
        String::from_utf8(out).unwrap()
    });

    Response::html(html.clone())
}
