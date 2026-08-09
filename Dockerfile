FROM alpine:3.20

RUN echo "https://dl-cdn.alpinelinux.org/alpine/v3.20/community" >> /etc/apk/repositories && \
    apk add --no-cache python3 py3-pip py3-requests git nodejs npm build-base make gcc musl-dev libffi-dev openssl-dev libedit-dev

# Build and install Trealla Prolog from source with musl compatibility patch and libedit dependencies
RUN git clone --depth 1 https://github.com/trealla-prolog/trealla.git /tmp/trealla && \
    cd /tmp/trealla && \
    sed -i '/_sigev_un/c\    #ifdef __GLIBC__\n    sevp._sigev_un._tid = syscall(SYS_gettid);\n    #endif' src/bif_os.c && \
    make && \
    cp tpl /usr/local/bin/ && \
    rm -rf /tmp/trealla

RUN WITH_EXTENSION=0 python3 -m pip install --no-cache-dir --break-system-packages mwparserfromhell
WORKDIR /INFO_SRC

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
    "@jax-js/jax": "^0.1.18",
    "@solidjs/meta": "^0.29.4",
    "@solidjs/router": "^0.15.3",
    "@solidjs/start": "^1.0.10",
    "solid-js": "^1.9.3",
    "vinxi": "^0.5.7"
  }
}
EOF

RUN npm install --legacy-peer-deps

RUN cat <<'EOF' > app.config.js
import { defineConfig } from "@solidjs/start/config";

export default defineConfig({
  server: {
    preset: "node-server",
    compressPublicAssets: false,
    experimental: {
      asyncContext: true
    }
  }
});
EOF

RUN mkdir -p src/routes && \
    cat <<'EOF' > src/entry-client.jsx
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
echo "Initializing services..."
echo "Starting Download Daemon on port 5000..."
python3 -u _download_INFO.py 2>&1 &
echo "Starting SolidStart Production Server on port 3000..."
HOST=0.0.0.0 PORT=3000 node .output/server/index.mjs 2>&1 &
wait -n
EOF

RUN chmod +x entrypoint.sh

RUN mkdir -p temp_facts && \
    git clone --depth 1 --filter=blob:none --sparse https://github.com/rahilshah13/FACTS.git temp_facts && \
    cd temp_facts && \
    git sparse-checkout set DICTIONARY/LANGUAGES/ENGLISH && \
    mv DICTIONARY/LANGUAGES/ENGLISH/predicates.pl ../ && \
    mv DICTIONARY/LANGUAGES/ENGLISH/words.pl ../ && \
    cd .. && rm -rf temp_facts

# Copy volatile application files
COPY index.jsx _download_INFO.py /INFO_SRC/

RUN cp index.jsx src/routes/index.jsx 2>/dev/null || cp src/index.jsx src/routes/index.jsx 2>/dev/null || true

# Force-clear cache, build, and link the public folder inside server/chunks for Nitro
RUN rm -rf .output .vinxi dist && \
    npm run build && \
    cd .output/server/chunks && \
    ln -s ../../public public

VOLUME [ "/info_txt_volume" ]
EXPOSE 3000
EXPOSE 5000
ENTRYPOINT ["/bin/sh", "entrypoint.sh"]