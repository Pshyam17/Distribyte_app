FROM node:18-alpine

WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Create public folder and copy HTML files
RUN mkdir -p public
COPY login.html public/
COPY dashboard.html public/
COPY 404.html public/

# Expose port
EXPOSE 3000

# Start the application
CMD ["npm", "start"]
