/*
world6_core.cpp — C++ port of World6 8×8 grid simulation.
==========================================================
Same interface as world6.py World6 class.
step(action) → (freq_hz, freq_idx, reward, info_dict)

Encoding: node "A0"=0, "A1"=1, ..., "H7"=63  (row*8+col)
Actions:  0=North, 1=East, 2=South, 3=West

Speed: ~30-50x faster than Python for the inner training loop.
*/

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <vector>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <string>
#include <cmath>
#include <random>
#include <algorithm>
#include <numeric>
#include <cassert>
#include <cstring>

namespace py = pybind11;
using namespace std;

// ── Constants ──────────────────────────────────────────────────────────────

static const int N_NODES     = 64;
static const int N_ACTIONS   = 4;   // N E S W
static const int N_ROWS      = 8;
static const int N_COLS      = 8;

// Actions: 0=North, 1=East, 2=South, 3=West
// ADJ[node][action] = neighbor index, or -1 if wall
static int ADJ[N_NODES][N_ACTIONS];

// Node properties
static float FREQ_HZ[N_NODES];
static int   FREQ_IDX[N_NODES];
static float TEXTURES[N_NODES];
static float ENV_VEG[N_NODES];
static float ENV_MOIST[N_NODES];
static float ENV_TEMP[N_NODES];
static float ENV_WIND[N_NODES];

// Node name strings "A0".."H7"
static char NODE_NAMES[N_NODES][3];

// inline helpers
static inline int node_idx(int r, int c) { return r * N_COLS + c; }
static inline bool in_set(const int* arr, int n, int v) {
    for (int i = 0; i < n; i++) if (arr[i] == v) return true;
    return false;
}

// Special node indices
// B2=10, D5=29, F1=41, H4=60
static const int BUTTON_NODES[] = {10, 29, 41, 60};
static const int N_BUTTON = 4;
// C7=23, E3=35, G6=54, A5=5
static const int DOOR_NODES[]   = {23, 35, 54, 5};
static const int N_DOOR = 4;
// C3=19, E6=38, G2=50, B6=14
static const int DANGER_NODES[] = {19, 38, 50, 14};
static const int N_DANGER = 4;
// A0=0, A7=7, H0=56, H7=63, A3=3, D0=24, D7=31, H3=59
static const int FOOD_CANDIDATES[] = {0, 7, 56, 63, 3, 24, 31, 59};
static const int N_FOOD_CANDS = 8;
// B0=8, C1=17, D2=26, E0=32, F1=41
static const int WATER_NODES[] = {8, 17, 26, 32, 41};
static const int N_WATER = 5;
// A0=0, A1=1, A2=2, B0=8, B1=9
static const int SOCIAL_NODES[] = {0, 1, 2, 8, 9};
static const int N_SOCIAL = 5;
// D3=27, D4=28, E3=35, E4=36
static const int REST_NODES[] = {27, 28, 35, 36};
static const int N_REST = 4;

static const int HOME_NODE     = 0;    // A0
static const int NPC_START     = 55;   // G7

// Drive constants
static const float HUNGER_RATE          = 0.020f;
static const float MAX_HUNGER           = 1.0f;
static const float HUNGER_MIN_MULT      = 0.10f;
static const float FATIGUE_RATE         = 0.004f;
static const float FATIGUE_WALL_BUMP    = 0.010f;
static const float FATIGUE_FOOD_RECOVER = 0.30f;
static const float THIRST_RATE          = 0.012f;
static const float MAX_THIRST           = 1.0f;
static const float THIRST_DRINK_RECOVER = 0.80f;
static const float THIRST_PENALTY       = -0.03f;
static const float THIRST_EXTREME       = 0.85f;
static const float INJURY_RATE          = 0.015f;
static const float INJURY_HEAL_RATE     = 0.005f;
static const float MAX_INJURY           = 1.0f;
static const float INJURY_HIGH          = 0.55f;
static const float INJURY_PAIN_MULT     = 1.5f;
static const float INJURY_REWARD        = 0.04f;

// Reward constants
static const float FOOD_REWARD          = 1.0f;
static const float WALL_PENALTY         = -0.05f;
static const float DANGER_PENALTY       = -0.04f;
static const float WATER_REWARD         = 0.6f;
static const float NPC_REWARD           = 0.05f;
static const float DISCOVERY_REWARD     = 0.08f;
static const float RECALL_REWARD        = 0.06f;
static const float SOCIAL_REWARD        = 0.02f;
static const float SATIATION_REWARD     = 0.05f;
static const float ANTICIPATION_REWARD  = 0.03f;
static const float REST_REWARD          = 0.04f;
static const float BOREDOM_REWARD       = -0.02f;
static const float SURPRISE_REWARD      = -0.02f;
static const float FRUSTRATION_REWARD   = -0.03f;
static const float STORM_REWARD         = -0.02f;

// Timing constants
static const int DOOR_OPEN_TICKS        = 15;
static const int FOOD_MOVE_INTERVAL     = 2000;
static const int N_FOOD_ACTIVE          = 2;
static const int NPC_MOVE_INTERVAL      = 40;
static const int RECALL_STEP_WINDOW     = 20;
static const int BOREDOM_WINDOW         = 50;
static const int BOREDOM_REVISIT_COUNT  = 8;
static const int FRUSTRATION_STREAK     = 4;
static const int STORM_DUR_MIN          = 80;
static const int STORM_DUR_MAX          = 200;
static const float STORM_PROB           = 0.0004f;
static const float STORM_FATIGUE_MULT   = 2.0f;


// ── Static table initialisation ────────────────────────────────────────────

static bool _tables_init = false;

static void init_tables() {
    if (_tables_init) return;
    _tables_init = true;

    // Node names
    for (int r = 0; r < N_ROWS; r++)
        for (int c = 0; c < N_COLS; c++) {
            int i = node_idx(r, c);
            NODE_NAMES[i][0] = 'A' + r;
            NODE_NAMES[i][1] = '0' + c;
            NODE_NAMES[i][2] = '\0';
        }

    // Adjacency: N=0, E=1, S=2, W=3
    for (int r = 0; r < N_ROWS; r++)
        for (int c = 0; c < N_COLS; c++) {
            int i = node_idx(r, c);
            ADJ[i][0] = (r > 0)           ? node_idx(r-1, c) : -1; // N
            ADJ[i][1] = (c < N_COLS-1)    ? node_idx(r, c+1) : -1; // E
            ADJ[i][2] = (r < N_ROWS-1)    ? node_idx(r+1, c) : -1; // S
            ADJ[i][3] = (c > 0)           ? node_idx(r, c-1) : -1; // W
        }

    // Frequencies: diagonal stripe — fi = (r + c) % 8
    static const float FREQS[8] = {0.5f,0.7f,0.9f,1.1f,1.3f,1.5f,1.7f,2.0f};
    for (int r = 0; r < N_ROWS; r++)
        for (int c = 0; c < N_COLS; c++) {
            int i  = node_idx(r, c);
            int fi = (r + c) % 8;
            FREQ_HZ[i]  = FREQS[fi];
            FREQ_IDX[i] = fi;
        }

    // Textures: within each fi group, rank nodes by (r+c) and assign linearly
    // tex_vals = linspace(0.05, 0.95, 8)
    static const float TEX_VALS[8] = {
        0.05f, 0.1786f, 0.3071f, 0.4357f, 0.5643f, 0.6929f, 0.8214f, 0.95f
    };
    // Group nodes by fi, sort by r+c, assign tex rank
    for (int fi = 0; fi < 8; fi++) {
        vector<pair<int,int>> group; // (r+c, node_idx)
        for (int r = 0; r < N_ROWS; r++)
            for (int c = 0; c < N_COLS; c++)
                if ((r+c) % 8 == fi)
                    group.push_back({r+c, node_idx(r,c)});
        sort(group.begin(), group.end());
        for (int k = 0; k < (int)group.size(); k++)
            TEXTURES[group[k].second] = TEX_VALS[k];
    }

    // Environmental profiles
    for (int r = 0; r < N_ROWS; r++)
        for (int c = 0; c < N_COLS; c++) {
            int i = node_idx(r, c);
            float dist_center = sqrtf(powf(r-3.5f,2)+powf(c-3.5f,2)) / 5.0f;
            float veg_row = 1.0f - fabsf(r-4.5f)/4.0f;
            float veg_col = 1.0f - fabsf(c-4.5f)/4.0f;
            ENV_VEG[i]   = fminf(fmaxf(veg_row*veg_col*1.8f, 0.05f), 1.0f);
            ENV_MOIST[i] = fminf(fmaxf(0.8f-c*0.08f+(1.0f-dist_center)*0.3f,0.05f),1.0f);
            ENV_TEMP[i]  = fminf(fmaxf(0.5f+(1.0f-dist_center)*0.4f-r*0.03f,0.1f),0.9f);
            ENV_WIND[i]  = fminf(fmaxf(dist_center*0.7f+(1.0f-ENV_VEG[i])*0.3f,0.05f),1.0f);
        }
}


// ── BFS optimal actions ─────────────────────────────────────────────────────

static unordered_map<int,int> bfs_from_targets(const vector<int>& targets) {
    vector<float> dist(N_NODES, 1e9f);
    deque<int> q;
    for (int t : targets) { dist[t] = 0; q.push_back(t); }
    while (!q.empty()) {
        int node = q.front(); q.pop_front();
        for (int a = 0; a < N_ACTIONS; a++) {
            int nbr = ADJ[node][a];
            if (nbr >= 0 && dist[nbr] > 1e8f) {
                dist[nbr] = dist[node] + 1;
                q.push_back(nbr);
            }
        }
    }
    unordered_map<int,int> opt;
    for (int node = 0; node < N_NODES; node++) {
        int best_a = 1;
        float best_d = 1e9f;
        for (int a = 0; a < N_ACTIONS; a++) {
            int nbr = ADJ[node][a];
            if (nbr >= 0 && dist[nbr] < best_d) {
                best_d = dist[nbr];
                best_a = a;
            }
        }
        opt[node] = best_a;
    }
    return opt;
}


// ── World6 C++ class ────────────────────────────────────────────────────────

class World6 {
public:
    // Drives
    float hunger, fatigue, thirst, injury;
    float time_of_day;
    int   node;           // current node index
    int   t;              // steps this episode
    int   total_steps;
    int   food_count, wall_count;

    // Button/door
    int door_timer;
    int button_step;

    // Dynamic food
    vector<int> active_food;
    int food_move_counter;
    bool food_moved_this_step;

    // Tracking
    unordered_set<int> discovered_nodes;
    unordered_set<int> prev_food_nodes;
    int  wall_streak;
    deque<int> recent_nodes;

    // NPC
    int npc_node;
    int npc_move_counter;

    // Storm
    int storm_ticks;

    // RNG
    mt19937 rng;
    uniform_real_distribution<float> uniform01;
    uniform_int_distribution<int>    rand_action;
    uniform_int_distribution<int>    rand_storm;

    explicit World6(int seed = 42)
        : hunger(0.5f), fatigue(0.2f), thirst(0.3f), injury(0.0f),
          time_of_day(0.5f), node(HOME_NODE), t(0), total_steps(0),
          food_count(0), wall_count(0), door_timer(0), button_step(-1),
          food_move_counter(0), food_moved_this_step(false),
          wall_streak(0), npc_node(NPC_START), npc_move_counter(0),
          storm_ticks(0), rng(seed), uniform01(0.0f, 1.0f),
          rand_action(0, N_ACTIONS-1),
          rand_storm(STORM_DUR_MIN, STORM_DUR_MAX-1)
    {
        init_tables();
        _pick_initial_food();
    }

    void _pick_initial_food() {
        active_food.clear();
        vector<int> cands(FOOD_CANDIDATES, FOOD_CANDIDATES + N_FOOD_CANDS);
        shuffle(cands.begin(), cands.end(), rng);
        for (int i = 0; i < N_FOOD_ACTIVE; i++)
            active_food.push_back(cands[i]);
    }

    py::tuple reset() {
        node = HOME_NODE;
        t = total_steps = food_count = wall_count = 0;
        hunger = 0.5f; fatigue = 0.2f; thirst = 0.3f; injury = 0.0f;
        time_of_day = 0.5f;
        door_timer = 0; button_step = -1;
        food_move_counter = 0; food_moved_this_step = false;
        wall_streak = 0;
        recent_nodes.clear();
        discovered_nodes.clear();
        prev_food_nodes.clear();
        npc_node = NPC_START; npc_move_counter = 0;
        storm_ticks = 0;
        _pick_initial_food();
        return py::make_tuple(FREQ_HZ[node], FREQ_IDX[node]);
    }

    py::tuple step(int action) {
        // ── Move ──────────────────────────────────────────────
        int next = ADJ[node][action];
        bool wall_hit = (next < 0);
        if (!wall_hit) node = next;

        t++;
        total_steps++;

        // ── Hunger ────────────────────────────────────────────
        hunger = fminf(MAX_HUNGER, hunger + HUNGER_RATE);

        // ── Fatigue + day/night ───────────────────────────────
        time_of_day = fmodf(time_of_day + 0.002f, 1.0f);
        bool is_night = (time_of_day > 0.75f || time_of_day < 0.25f);
        float night_mult = is_night ? 1.5f : 1.0f;
        if (wall_hit)
            fatigue = fminf(1.0f, fatigue + (FATIGUE_RATE + FATIGUE_WALL_BUMP) * night_mult);
        else
            fatigue = fminf(1.0f, fatigue + FATIGUE_RATE * night_mult);

        // ── Thirst ────────────────────────────────────────────
        thirst = fminf(MAX_THIRST, thirst + THIRST_RATE);

        // ── Injury ────────────────────────────────────────────
        bool is_danger = !wall_hit && in_set(DANGER_NODES, N_DANGER, node);
        if (!wall_hit && is_danger)
            injury = fminf(MAX_INJURY, injury + INJURY_RATE);
        else if (injury > 0.0f)
            injury = fmaxf(0.0f, injury - INJURY_HEAL_RATE);

        // ── Storm ─────────────────────────────────────────────
        if (storm_ticks > 0)
            storm_ticks--;
        else if (uniform01(rng) < STORM_PROB)
            storm_ticks = rand_storm(rng);
        bool is_storm = (storm_ticks > 0);

        // ── NPC wander ────────────────────────────────────────
        npc_move_counter++;
        if (npc_move_counter >= NPC_MOVE_INTERVAL) {
            npc_move_counter = 0;
            vector<int> npc_nbrs;
            for (int a = 0; a < N_ACTIONS; a++) {
                int nb = ADJ[npc_node][a];
                if (nb >= 0) npc_nbrs.push_back(nb);
            }
            if (!npc_nbrs.empty()) {
                uniform_int_distribution<int> rn(0, (int)npc_nbrs.size()-1);
                npc_node = npc_nbrs[rn(rng)];
            }
        }

        // ── Door timer ────────────────────────────────────────
        if (door_timer > 0) door_timer--;

        // ── Discovery ─────────────────────────────────────────
        bool is_novel = !wall_hit && (discovered_nodes.find(node) == discovered_nodes.end());
        if (is_novel) discovered_nodes.insert(node);

        // ── Wall streak ───────────────────────────────────────
        if (wall_hit) wall_streak++;
        else          wall_streak = 0;

        // ── Boredom tracking ──────────────────────────────────
        if (!wall_hit) {
            recent_nodes.push_back(node);
            if ((int)recent_nodes.size() > BOREDOM_WINDOW)
                recent_nodes.pop_front();
        }

        // ── Button ────────────────────────────────────────────
        bool is_button = false;
        if (!wall_hit && in_set(BUTTON_NODES, N_BUTTON, node)) {
            is_button  = true;
            door_timer = DOOR_OPEN_TICKS;
            button_step = t;
        }

        // ── Rewards ───────────────────────────────────────────
        float reward = 0.0f;
        bool is_food=false, is_door=false, is_recall=false, is_social=false;
        bool is_satiated=false, is_frustrated=false, is_anticipating=false;
        bool is_resting=false, is_bored=false, is_surprise=false;
        bool is_drinking=false, is_healing=false, is_npc_meet=false;

        if (wall_hit) {
            reward = WALL_PENALTY;
            wall_count++;
        } else {
            // Danger
            if (is_danger) {
                float pm = (injury > INJURY_HIGH) ? INJURY_PAIN_MULT : 1.0f;
                reward = DANGER_PENALTY * pm;
            }

            // Water
            if (in_set(WATER_NODES, N_WATER, node)) {
                is_drinking = true;
                float tm = fmaxf(0.1f, thirst);
                reward += WATER_REWARD * tm;
                thirst = fmaxf(0.0f, thirst - THIRST_DRINK_RECOVER);
            }

            // Injury heal reward
            if (injury < 0.2f && injury + INJURY_HEAL_RATE >= 0.2f) {
                reward += INJURY_REWARD;
                is_healing = true;
            }

            // NPC meet: adjacent or same node
            bool npc_adj = false;
            for (int a = 0; a < N_ACTIONS; a++)
                if (ADJ[npc_node][a] == node) { npc_adj = true; break; }
            if (node == npc_node || npc_adj) {
                reward += NPC_REWARD;
                is_npc_meet = true;
            }

            // Door
            if (in_set(DOOR_NODES, N_DOOR, node)) {
                is_door = true;
                if (door_timer > 0) {
                    float hm = fmaxf(HUNGER_MIN_MULT, hunger);
                    reward = FOOD_REWARD * hm;
                    is_food = true;
                    hunger  = 0.0f;
                    fatigue = fmaxf(0.0f, fatigue - FATIGUE_FOOD_RECOVER);
                    food_count++;
                    if (button_step >= 0 && t - button_step <= RECALL_STEP_WINDOW) {
                        reward += RECALL_REWARD;
                        is_recall = true;
                    }
                }
            }
            // Direct food nodes
            else if (find(active_food.begin(), active_food.end(), node) != active_food.end()) {
                float hm = fmaxf(HUNGER_MIN_MULT, hunger);
                reward = FOOD_REWARD * hm;
                is_food  = true;
                hunger   = 0.0f;
                fatigue  = fmaxf(0.0f, fatigue - FATIGUE_FOOD_RECOVER);
                food_count++;
            }

            // Discovery
            if (is_novel) reward += DISCOVERY_REWARD;

            // Satiation
            if (is_food) { reward += SATIATION_REWARD; is_satiated = true; }

            // Anticipation
            if (door_timer > 0 && !is_door) {
                is_anticipating = true;
                reward += ANTICIPATION_REWARD;
            }

            // Rest
            if (in_set(REST_NODES, N_REST, node) && fatigue > 0.5f && is_night) {
                reward += REST_REWARD;
                is_resting = true;
            }

            // Social
            if (in_set(SOCIAL_NODES, N_SOCIAL, node)) {
                reward += SOCIAL_REWARD;
                is_social = true;
            }
        }

        // Storm extra fatigue
        if (is_storm) {
            fatigue = fminf(1.0f, fatigue + FATIGUE_RATE * (STORM_FATIGUE_MULT - 1.0f));
            reward += STORM_REWARD;
        }

        // Extreme thirst
        if (thirst > THIRST_EXTREME) reward += THIRST_PENALTY;

        // Frustration
        if (wall_streak >= FRUSTRATION_STREAK) {
            reward += FRUSTRATION_REWARD;
            is_frustrated = true;
        }

        // Boredom
        if ((int)recent_nodes.size() >= BOREDOM_WINDOW) {
            int cnt = 0;
            for (int n : recent_nodes) if (n == node) cnt++;
            if (cnt >= BOREDOM_REVISIT_COUNT) {
                reward += BOREDOM_REWARD;
                is_bored = true;
            }
        }

        // Surprise (prediction error)
        if (!wall_hit && !is_food) {
            if (in_set(DOOR_NODES, N_DOOR, node) && door_timer == 0)
                is_surprise = true;
            else if (prev_food_nodes.count(node))
                is_surprise = true;
        }
        if (is_surprise) reward += SURPRISE_REWARD;

        // ── Food rotation ──────────────────────────────────────
        food_moved_this_step = false;
        food_move_counter++;
        if (food_move_counter >= FOOD_MOVE_INTERVAL) {
            food_move_counter = 0;
            _rotate_food();
            food_moved_this_step = true;
        }

        // ── Build info dict ────────────────────────────────────
        py::dict info;
        info["node"]          = string(NODE_NAMES[node]);
        info["label"]         = string(NODE_NAMES[node]);
        info["wall_hit"]      = wall_hit;
        info["is_food"]       = is_food;
        info["is_button"]     = is_button;
        info["is_door"]       = is_door;
        info["is_danger"]     = is_danger;
        info["door_open"]     = (door_timer > 0);
        info["door_ticks"]    = door_timer;
        info["texture"]       = TEXTURES[node];
        info["hunger"]        = hunger;
        info["fatigue"]       = fatigue;
        info["food_moved"]    = food_moved_this_step;
        info["food_nodes"]    = _active_food_names();
        info["time_of_day"]   = time_of_day;
        info["is_night"]      = is_night;
        info["is_novel"]      = is_novel;
        info["is_recall"]     = is_recall;
        info["is_social"]     = is_social;
        info["is_satiated"]   = is_satiated;
        info["is_frustrated"] = is_frustrated;
        info["is_anticipating"] = is_anticipating;
        info["is_resting"]    = is_resting;
        info["is_bored"]      = is_bored;
        info["is_surprise"]   = is_surprise;
        info["thirst"]        = thirst;
        info["injury"]        = injury;
        info["is_drinking"]   = is_drinking;
        info["is_healing"]    = is_healing;
        info["is_npc_meet"]   = is_npc_meet;
        info["is_storm"]      = is_storm;
        info["npc_node"]      = string(NODE_NAMES[npc_node]);
        // Environmental profile
        py::dict env;
        env["vegetation"]  = ENV_VEG[node];
        env["moisture"]    = ENV_MOIST[node];
        env["temperature"] = ENV_TEMP[node];
        env["wind"]        = ENV_WIND[node];
        info["env"]        = env;

        return py::make_tuple(FREQ_HZ[node], FREQ_IDX[node], reward, info);
    }

    // ── Helpers ───────────────────────────────────────────────

    void _rotate_food() {
        // Collect candidates not currently active
        vector<int> available;
        for (int i = 0; i < N_FOOD_CANDS; i++) {
            int c = FOOD_CANDIDATES[i];
            if (find(active_food.begin(), active_food.end(), c) == active_food.end())
                available.push_back(c);
        }
        if (available.empty()) return;
        int retiring = active_food.front();
        active_food.erase(active_food.begin());
        prev_food_nodes.insert(retiring);
        uniform_int_distribution<int> ri(0, (int)available.size()-1);
        int new_food = available[ri(rng)];
        active_food.push_back(new_food);
        prev_food_nodes.erase(new_food);
    }

    vector<string> _active_food_names() const {
        vector<string> v;
        for (int f : active_food) v.push_back(string(NODE_NAMES[f]));
        return v;
    }

    // ── Python-visible properties ──────────────────────────────

    string get_current_node() const { return string(NODE_NAMES[node]); }
    float  get_hunger()       const { return hunger; }
    float  get_thirst()       const { return thirst; }
    float  get_fatigue()      const { return fatigue; }

    vector<string> get_active_food() const { return _active_food_names(); }

    float food_rate() const {
        return total_steps == 0 ? 0.0f : float(food_count) / float(total_steps) * 100.0f;
    }
    float wall_rate() const {
        return total_steps == 0 ? 0.0f : float(wall_count) / float(total_steps);
    }
};


// ── BFS helper exposed to Python ───────────────────────────────────────────

static py::dict py_bfs_optimal(const vector<string>& food_strs,
                                const vector<string>& button_strs,
                                const vector<string>& door_strs)
{
    init_tables();

    auto name2idx = [](const string& s) -> int {
        if (s.size() < 2) return -1;
        int r = s[0] - 'A';
        int c = s[1] - '0';
        if (r < 0 || r >= N_ROWS || c < 0 || c >= N_COLS) return -1;
        return node_idx(r, c);
    };

    vector<int> targets;
    for (auto& s : door_strs)   { int i = name2idx(s); if (i>=0) targets.push_back(i); }
    for (auto& s : food_strs)   { int i = name2idx(s); if (i>=0) targets.push_back(i); }

    auto opt = bfs_from_targets(targets);

    py::dict result;
    for (int n = 0; n < N_NODES; n++)
        result[py::str(NODE_NAMES[n])] = opt[n];
    return result;
}

// BFS to arbitrary target set (for water policy)
static py::dict py_bfs_to_targets(const vector<string>& target_strs)
{
    init_tables();
    auto name2idx = [](const string& s) -> int {
        if (s.size() < 2) return -1;
        int r = s[0] - 'A', c = s[1] - '0';
        return (r>=0&&r<N_ROWS&&c>=0&&c<N_COLS) ? node_idx(r,c) : -1;
    };
    vector<int> targets;
    for (auto& s : target_strs) { int i = name2idx(s); if (i>=0) targets.push_back(i); }
    auto opt = bfs_from_targets(targets);
    py::dict result;
    for (int n = 0; n < N_NODES; n++)
        result[py::str(NODE_NAMES[n])] = opt[n];
    return result;
}


// ── Module ─────────────────────────────────────────────────────────────────

PYBIND11_MODULE(world6_core, m) {
    m.doc() = "World6 C++ simulation — 30-50x faster than Python equivalent";

    init_tables();

    py::class_<World6>(m, "World6")
        .def(py::init<int>(), py::arg("seed")=42)
        .def("reset",    &World6::reset)
        .def("step",     &World6::step)
        .def_property_readonly("current_node",  &World6::get_current_node)
        .def_property_readonly("_hunger",       &World6::get_hunger)
        .def_property_readonly("_thirst",       &World6::get_thirst)
        .def_property_readonly("_fatigue",      &World6::get_fatigue)
        .def_property_readonly("_active_food",  &World6::get_active_food)
        .def("food_rate", &World6::food_rate)
        .def("wall_rate", &World6::wall_rate);

    m.def("bfs_optimal_actions", &py_bfs_optimal,
          py::arg("food_nodes"), py::arg("button_nodes"), py::arg("door_nodes"));
    m.def("bfs_to_targets", &py_bfs_to_targets,
          py::arg("target_nodes"));
}
