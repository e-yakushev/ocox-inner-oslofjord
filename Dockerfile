FROM europe-west1-docker.pkg.dev/niva-cd/images/fjordsim-oceananigans:ef91103

WORKDIR /app

COPY simulation.jl ./simulation.jl
COPY Oxydep.jl ./Oxydep.jl
COPY scenarios.jl ./scenarios.jl
COPY scenarios.json ./scenarios.json

ENV SIMULATION_LAUNCHER=/app/simulation.jl
ENV PROJECT_ROOT=/app
ENV JULIA_DEPOT_PATH=/usr/local/julia_depot
ENV JULIA_PROJECT=/app

COPY entrypoint.sh .
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]