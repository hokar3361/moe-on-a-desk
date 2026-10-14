# Stop rule (LLAMA_SPEC_STOP=1) parameters per model: LLAMA_SPEC_STOP_PARAMS=<file> loads the logistic models (fit_s1.py output),
# q̄, the verify time by width and the draft-step cost, so another model (e.g. Qwen3.6-35B-A3B) gets its own rule instead of the
# Qwen3.8-Flash-Next constants. File format: one array per line, "<name> v1 v2 ...", names mu_post sd_post w_post mu_pre sd_pre
# w_pre qbar (positions 1..) tv (widths 0..) c_draft lam. Missing names keep the built-in values.
import sys
p = sys.argv[1] + "/common/speculative.cpp"
s = open(p, encoding="utf-8").read()
old = """            if (const char * e = getenv("LLAMA_SPEC_STOP_LAM")) sr.lam = (float) atof(e);"""
assert s.count(old) == 1
new = """            if (const char * f = getenv("LLAMA_SPEC_STOP_PARAMS")) sr.load(f);
            if (const char * e = getenv("LLAMA_SPEC_STOP_LAM")) sr.lam = (float) atof(e);"""
s = s.replace(old, new)
old2 = """        void reset_request() {
            for (int k = 0; k < 18; ++k) ema[k] = 0.7f;"""
assert s.count(old2) == 1
new2 = """        // per-model parameters (LLAMA_SPEC_STOP_PARAMS): "<name> v1 v2 ..." per line; qbar from position 1, tv from width 0
        void load(const char * path) {
            FILE * fp = fopen(path, "r");
            if (!fp) { LOG_WRN("stop rule: cannot read %s, keeping the built-in parameters@NL@", path); return; }
            char line[4096];
            int n_set = 0;
            while (fgets(line, sizeof(line), fp)) {
                char name[32];
                int off = 0;
                if (sscanf(line, "%31s%n", name, &off) != 1) continue;
                float v[32];
                int n = 0, used = 0;
                const char * q = line + off;
                while (n < 32 && sscanf(q, "%f%n", &v[n], &used) == 1) { q += used; ++n; }
                auto put = [&](float * dst, int cap, int first) { for (int i = 0; i < n && first + i < cap; ++i) dst[first + i] = v[i]; ++n_set; };
                const std::string nm = name;
                if      (nm == "mu_post") put(mu_post, NF, 0);
                else if (nm == "sd_post") put(sd_post, NF, 0);
                else if (nm == "w_post")  put(w_post, NF + 1, 0);
                else if (nm == "mu_pre")  put(mu_pre, NF, 0);
                else if (nm == "sd_pre")  put(sd_pre, NF, 0);
                else if (nm == "w_pre")   put(w_pre, NF + 1, 0);
                else if (nm == "qbar")    put(qbar, 18, 1);
                else if (nm == "tv")      put(tv, 18, 0);
                else if (nm == "c_draft" && n > 0) { c_draft = v[0]; ++n_set; }
                else if (nm == "lam" && n > 0)     { lam = v[0]; ++n_set; }
            }
            fclose(fp);
            LOG_INF("stop rule: %d parameter sets from %s (draft %.2f ms, verify w=4 %.2f ms)@NL@", n_set, path, c_draft, tv[4]);
        }
        void reset_request() {
            for (int k = 0; k < 18; ++k) ema[k] = 0.7f;"""
s = s.replace(old2, new2.replace("@NL@", chr(92) + "n"))
open(p, "w", encoding="utf-8", newline=chr(10)).write(s)
print("ok")
