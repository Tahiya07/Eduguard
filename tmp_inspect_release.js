const fs = require("fs");
const path = "release/app/frontend/.next/static/chunks/914-a6c4d899c65a67ab.js";
const s = fs.readFileSync(path, "utf8");
const terms = ["teacher/exam/moderate", "teacher/exam/classify", "Invalid", "Bearer"];
for (const t of terms) {
  const i = s.indexOf(t);
  console.log("\n===", t, "at", i, "===");
  if (i >= 0) console.log(s.substring(Math.max(0, i - 250), i + 350));
}
