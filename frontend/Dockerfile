FROM node:20-slim

WORKDIR /app

COPY package*.json ./

# Force optional deps (Rollup fix) + local Vite install
RUN npm cache clean --force && \
    npm install --include=optional

COPY . .

EXPOSE 5173

# IMPORTANT: use npm script, not npx
CMD ["npm", "run", "dev", "--", "--host"]
