const fs = require("fs");
const s = fs.readFileSync("frontend/.next/static/chunks/914-90ad8167c21d19f4.js", "utf8");
const i = s.indexOf("teacher/exam/moderate");
const chunk = s.substring(Math.max(0, i - 900), i + 200);
console.log(chunk);
const apiMatch = s.match(/localhost:8000|127\.0\.0\.1:8000/);
console.log("\nAPI match:", apiMatch && apiMatch[0]);
