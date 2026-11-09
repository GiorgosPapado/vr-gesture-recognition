## Usage

1. use NPM package `light-server` to serve the current folder at http://localhost:8080/
   ```sh
   
   cd project location
   npx light-server -s . -p 8080
   ```

2. open your browser and navigate to the URL.

## Sample data

`gesture_data_train.json` and `gesture_data_test.json` are not duplicated here; `JS_Integration.html` loads them from `../webxr/`, the single source. Serve the repo root (not this folder alone) so that relative path resolves.
