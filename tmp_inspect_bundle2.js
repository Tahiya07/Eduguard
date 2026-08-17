const fs = require("fs");
const s = fs.readFileSync("frontend/.next/static/chunks/914-90ad8167c21d19f4.js", "utf8");
// find module start - look for localhost:8000 assignment
const idx = s.indexOf("localhost:8000");
console.log(s.substring(Math.max(0, idx - 200), idx + 100));
