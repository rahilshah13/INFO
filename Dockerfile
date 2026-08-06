FROM alpine:3.20

RUN apk add --no-cache python3 py3-pip py3-requests git nodejs npm
RUN WITH_EXTENSION=0 python3 -m pip install --no-cache-dir --break-system-packages mwparserfromhell

COPY . /INFO_SRC
WORKDIR /INFO_SRC

RUN rm -rf .output .vinxi node_modules dist

RUN cat <<'EOF' > package.json
{
  "name": "solid-jax-info-app",
  "type": "module",
  "scripts": {
    "dev": "vinxi dev",
    "build": "vinxi build",
    "start": "vinxi start"
  },
  "dependencies": {
    "@solidjs/meta": "^0.29.4",
    "@solidjs/router": "^0.15.3",
    "@solidjs/start": "^1.0.10",
    "solid-js": "^1.9.3",
    "vinxi": "^0.5.7"
  }
}
EOF

RUN cat <<'EOF' > app.config.js
import { defineConfig } from "@solidjs/start/config";

export default defineConfig({
  server: {
    compressPublicAssets: false,
    experimental: {
      asyncContext: true
    }
  }
});
EOF

RUN mkdir -p src/routes
RUN cp index.jsx src/routes/index.jsx 2>/dev/null || cp src/index.jsx src/routes/index.jsx 2>/dev/null || true

RUN cat <<'EOF' > src/entry-client.jsx
import { mount, StartClient } from "@solidjs/start/client";

mount(() => <StartClient />, document.getElementById("app"));
EOF

RUN cat <<'EOF' > src/entry-server.jsx
import { createHandler, StartServer } from "@solidjs/start/server";

export default createHandler(() => (
  <StartServer
    document={({ assets, children, scripts }) => (
      <html lang="en">
        <head>
          <meta charset="utf-8" />
          <meta name="viewport" content="width=device-width, initial-scale=1" />
          {assets}
        </head>
        <body>
          <div id="app">{children}</div>
          {scripts}
        </body>
      </html>
    )}
  />
));
EOF

RUN cat <<'EOF' > src/app.jsx
import { Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import { Suspense } from "solid-js";

export default function App() {
  return (
    <Router
      root={props => (
        <Suspense>{props.children}</Suspense>
      )}
    >
      <FileRoutes />
    </Router>
  );
}
EOF

RUN cat <<'EOF' > entrypoint.sh
#!/bin/sh
set -e

echo "Starting Download Daemon on port 5000..."
python3 _download_INFO.py &

echo "Starting SolidStart JAX-JS Web Server on port 3000..."
npm run start &

wait -n
EOF

RUN chmod +x entrypoint.sh

RUN npm install --legacy-peer-deps
RUN npm run build

RUN mkdir -p .output/server/chunks && ln -s /INFO_SRC/.output/public .output/server/chunks/public

RUN mkdir -p temp_facts && \
    git clone --depth 1 --filter=blob:none --sparse https://github.com/rahilshah13/FACTS.git temp_facts && \
    cd temp_facts && \
    git sparse-checkout set DICTIONARY/LANGUAGES/ENGLISH && \
    mv DICTIONARY/LANGUAGES/ENGLISH/predicates.pl ../ && \
    mv DICTIONARY/LANGUAGES/ENGLISH/words.pl ../ && \
    cd .. && rm -rf temp_facts

VOLUME [ "/info_txt_volume" ]

EXPOSE 3000
EXPOSE 5000

ENTRYPOINT ["/bin/sh", "entrypoint.sh"]