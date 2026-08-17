const fs = require("fs");
const path = "release/app/frontend/.next/static/chunks/914-a6c4d899c65a67ab.js";
const s = fs.readFileSync(path, "utf8");
const m = s.match(/let n=[^,]+,/);
console.log(m && m[0]);
