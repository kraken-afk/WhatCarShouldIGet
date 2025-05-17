use dioxus::prelude::*;
#[derive(Debug, Clone, Routable, PartialEq)]
#[rustfmt::skip]
enum Route {
    #[route("/")]
    Home {},
}
const FAVICON: Asset = asset!("/assets/favicon.ico");
const MAIN_CSS: Asset = asset!("/assets/main.css");
fn main() {
    dioxus::launch(App);
}
#[component]
fn App() -> Element {
    rsx! {
        document::Link { rel: "icon", href: FAVICON }
        document::Link { rel: "stylesheet", href: MAIN_CSS }
        Router::<Route> {}
    }
}
/// Home page
#[component]
fn Home() -> Element {
    let mut count = use_signal(|| 0);
    rsx! {
        div { class: "w-full h-dvh flex items-center justify-center",
            div { class: "p-4 flex flex-col space-y-4 items-center",
                h1 { class: "text-4xl font-bold", "{count}" }
                div { class: "flex space-x-4",
                    button { onclick: move |_| count += 1,
                        span { class: "text-teal-400 border rounded-sm py-1 px-2", "Incerement" }
                    }
                    button { onclick: move |_| count -= 1,
                        span { class: "text-teal-400 border rounded-sm py-1 px-2", "Decrement" }
                    }
                }
            }
        }
    }
}
/// Echo the user input on the server.
#[server(EchoServer)]
async fn echo_server(input: String) -> Result<String, ServerFnError> {
    Ok(input)
}
