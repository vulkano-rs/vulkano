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
use std::collections::hash_map::Entry;
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
                        main_template(include_str!("../content/home.html"))
                    },
                    _ => {
                        if let Some(request) = request.remove_prefix("/guides") {
                            if request.raw_url().starts_with("/01") {
                                guide_template_markdown(include_str!("../content/01-getting-started.md"))
                            } else if request.raw_url().starts_with("/02") {
                                guide_template_markdown(include_str!("../content/02-first-operation.md"))
                            } else if request.raw_url().starts_with("/03") {
                                guide_template_markdown(include_str!("../content/03-window-swapchain.md"))
                            } else if request.raw_url().starts_with("/04") {
                                guide_template_markdown(include_str!("../content/04-render-pass.md"))
                            } else if request.raw_url().starts_with("/05") {
                                guide_template_markdown(include_str!("../content/05-first-triangle.md"))
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

fn main_template<S>(body: S) -> Response
    where S: Into<String>
{
    lazy_static! {
        static ref MAIN_TEMPLATE: mustache::Template = {
            mustache::compile_str(&include_str!("../content/template_main.html")).unwrap()
        };

        static ref CACHE: Mutex<HashMap<String, String>> = Mutex::new(HashMap::new());
    }

    let body = body.into();

    let mut compil_cache = CACHE.lock().unwrap();
    let html = match compil_cache.entry(body) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let data = mustache::MapBuilder::new()
                .insert_str("body", e.key())
                .build();

            let mut out = Vec::new();
            MAIN_TEMPLATE.render_data(&mut out, &data).unwrap();
            e.insert(String::from_utf8(out).unwrap())
        }
    };

    Response::html(html.clone())
}

fn guide_template<S>(body: S) -> Response
    where S: Into<String>
{
    lazy_static! {
        static ref GUIDE_TEMPLATE: mustache::Template = {
            mustache::compile_str(&include_str!("../content/template_guide.html")).unwrap()
        };

        static ref CACHE: Mutex<HashMap<String, String>> = Mutex::new(HashMap::new());
    }

    let body = body.into();

    let mut compil_cache = CACHE.lock().unwrap();
    let html = match compil_cache.entry(body) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let data = mustache::MapBuilder::new()
                .insert_str("body", e.key())
                .build();

            let mut out = Vec::new();
            GUIDE_TEMPLATE.render_data(&mut out, &data).unwrap();
            e.insert(String::from_utf8(out).unwrap())
        }
    };

    main_template(html.clone())
}

fn guide_template_markdown<S>(body: S) -> Response
    where S: Into<String>
{
    lazy_static! {
        static ref CACHE: Mutex<HashMap<String, String>> = Mutex::new(HashMap::new());
    }

    let body = body.into();

    let mut compil_cache = CACHE.lock().unwrap();
    let html = match compil_cache.entry(body) {
        Entry::Occupied(e) => e.into_mut(),
        Entry::Vacant(e) => {
            let val = markdown::to_html(e.key());
            e.insert(val)
        },
    };

    guide_template(html.clone())
}
