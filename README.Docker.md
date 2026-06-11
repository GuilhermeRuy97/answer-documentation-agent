### Building and running your application

Prerequisite: a filled-in `.env` file in the project root (compose passes it
to the container via `env_file`).

When you're ready, start your application by running:
`docker compose up --build`.

Your application (API + chat UI) will be available at http://localhost:8000.
The container healthcheck polls `GET /health`, which is intentionally public.

### Deploying your application to the cloud

First, build your image, e.g.: `docker build -t myapp .`.
If your cloud uses a different CPU architecture than your development
machine (e.g., you are on a Mac M1 and your cloud provider is amd64),
you'll want to build the image for that platform, e.g.:
`docker build --platform=linux/amd64 -t myapp .`.

Then, push it to your registry, e.g. `docker push myregistry.com/myapp`.

Consult Docker's [getting started](https://docs.docker.com/go/get-started-sharing/)
docs for more detail on building and pushing.

### References
* [Docker's Python guide](https://docs.docker.com/language/python/)