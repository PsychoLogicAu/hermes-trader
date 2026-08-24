# Docker Compose Setup (if you install Docker)

# Create the network (required by the stack)
docker network create llama-net

# Configure files
cp .env.example .env.local
nano .env.local

cp .agent-config.example.json .agent-config.json
nano .agent-config.json

# Host-side volume dirs must be owned by the same uid/gid the image is built
# with (scripts/build.sh detects them from the host automatically). If any
# dir is missing, Docker creates it as root:root and the container crashes
# with PermissionError.
mkdir -p trader-logs hf-cache agent-state
sudo chown -R "$(id -u):$(id -g)" trader-logs hf-cache agent-state

# Start (builds on first run — build.sh matches the container user to this
# host's uid/gid; plain `docker compose up` builds with 1000:1000)
scripts/build.sh
docker compose up -d

# Monitor
docker compose logs -f hermes-trader
docker compose ps
