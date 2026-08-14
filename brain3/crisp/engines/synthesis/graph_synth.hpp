#pragma once
#include <map>
#include <vector>
#include <set>
#include <queue>
#include <stack>
#include <string>
#include <functional>
#include <iostream>

namespace brain3 {
namespace engines {
namespace synthesis {

using AdjList = std::map<int, std::vector<int>>;
using WAdjList = std::map<int, std::vector<std::pair<int, double>>>;

class GraphSynth {
public:
    static std::map<int, int> bfs_dist(const AdjList& adj, int src) {
        std::map<int, int> dist;
        dist[src] = 0;
        std::queue<int> q;
        q.push(src);
        while (!q.empty()) {
            int node = q.front(); q.pop();
            auto it = adj.find(node);
            if (it != adj.end()) {
                for (int nb : it->second) {
                    if (dist.find(nb) == dist.end()) {
                        dist[nb] = dist[node] + 1;
                        q.push(nb);
                    }
                }
            }
        }
        return dist;
    }

    static std::set<int> bfs_reach(const AdjList& adj, int src) {
        std::set<int> seen;
        seen.insert(src);
        std::queue<int> q;
        q.push(src);
        while (!q.empty()) {
            int node = q.front(); q.pop();
            auto it = adj.find(node);
            if (it != adj.end()) {
                for (int nb : it->second) {
                    if (seen.find(nb) == seen.end()) {
                        seen.insert(nb);
                        q.push(nb);
                    }
                }
            }
        }
        return seen;
    }

    static std::set<int> dfs_reach(const AdjList& adj, int src) {
        std::set<int> seen;
        seen.insert(src);
        std::vector<int> stack;
        stack.push_back(src);
        while (!stack.empty()) {
            int node = stack.back(); stack.pop_back();
            auto it = adj.find(node);
            if (it != adj.end()) {
                for (int nb : it->second) {
                    if (seen.find(nb) == seen.end()) {
                        seen.insert(nb);
                        stack.push_back(nb);
                    }
                }
            }
        }
        return seen;
    }

    static bool has_path(const AdjList& adj, int src, int dst) {
        std::set<int> r = bfs_reach(adj, src);
        return r.find(dst) != r.end();
    }

    static int connected_components(const AdjList& adj) {
        std::set<int> seen;
        int count = 0;
        for (const auto& [node, nbs] : adj) {
            if (seen.find(node) == seen.end()) {
                std::set<int> comp = bfs_reach(adj, node);
                seen.insert(comp.begin(), comp.end());
                count++;
            }
        }
        return count;
    }

    static std::vector<int> topo_order(const AdjList& adj) {
        std::map<int, int> in_deg;
        for (const auto& [n, nbs] : adj) {
            if (in_deg.find(n) == in_deg.end()) in_deg[n] = 0;
            for (int nb : nbs) in_deg[nb]++;
        }
        std::queue<int> q;
        for (const auto& [n, d] : in_deg) {
            if (d == 0) q.push(n);
        }
        std::vector<int> order;
        while (!q.empty()) {
            int n = q.front(); q.pop();
            order.push_back(n);
            auto it = adj.find(n);
            if (it != adj.end()) {
                for (int nb : it->second) {
                    in_deg[nb]--;
                    if (in_deg[nb] == 0) q.push(nb);
                }
            }
        }
        if (order.size() == adj.size()) return order;
        return {};
    }

    static std::map<int, double> dijkstra(const WAdjList& wadj, int src) {
        std::map<int, double> dist;
        dist[src] = 0.0;
        using PDI = std::pair<double, int>;
        std::priority_queue<PDI, std::vector<PDI>, std::greater<PDI>> heap;
        heap.push({0.0, src});
        
        while (!heap.empty()) {
            auto [cost, node] = heap.top(); heap.pop();
            if (dist.find(node) != dist.end() && cost > dist[node]) continue;
            auto it = wadj.find(node);
            if (it != wadj.end()) {
                for (const auto& [nb, w] : it->second) {
                    double nc = cost + w;
                    if (dist.find(nb) == dist.end() || nc < dist[nb]) {
                        dist[nb] = nc;
                        heap.push({nc, nb});
                    }
                }
            }
        }
        return dist;
    }

    // Synthesis API for different schemas
    std::pair<std::string, std::string> synthesize_adj_src_map(
        const std::vector<std::pair<std::pair<AdjList, int>, std::map<int, int>>>& examples) {
        
        // Try bfs_dist
        bool ok = true;
        for (const auto& ex : examples) {
            if (bfs_dist(ex.first.first, ex.first.second) != ex.second) { ok = false; break; }
        }
        if (ok) return {"bfs_dist", "code_for_bfs_dist"};
        
        return {"", ""};
    }
    
    std::pair<std::string, std::string> synthesize_adj_src_set(
        const std::vector<std::pair<std::pair<AdjList, int>, std::set<int>>>& examples) {
        
        bool ok_bfs = true, ok_dfs = true;
        for (const auto& ex : examples) {
            if (bfs_reach(ex.first.first, ex.first.second) != ex.second) ok_bfs = false;
            if (dfs_reach(ex.first.first, ex.first.second) != ex.second) ok_dfs = false;
        }
        if (ok_bfs) return {"bfs_reach", "code_for_bfs_reach"};
        if (ok_dfs) return {"dfs_reach", "code_for_dfs_reach"};
        return {"", ""};
    }
    
    std::pair<std::string, std::string> synthesize_adj_src_dst_bool(
        const std::vector<std::pair<std::tuple<AdjList, int, int>, bool>>& examples) {
        bool ok = true;
        for (const auto& ex : examples) {
            auto [adj, src, dst] = ex.first;
            if (has_path(adj, src, dst) != ex.second) { ok = false; break; }
        }
        if (ok) return {"has_path", "code_for_has_path"};
        return {"", ""};
    }
    
    std::pair<std::string, std::string> synthesize_adj_int(
        const std::vector<std::pair<AdjList, int>>& examples) {
        bool ok = true;
        for (const auto& ex : examples) {
            if (connected_components(ex.first) != ex.second) { ok = false; break; }
        }
        if (ok) return {"connected_components", "code_for_connected_components"};
        return {"", ""};
    }

    std::pair<std::string, std::string> synthesize_adj_list(
        const std::vector<std::pair<AdjList, std::vector<int>>>& examples) {
        bool ok = true;
        for (const auto& ex : examples) {
            if (topo_order(ex.first) != ex.second) { ok = false; break; }
        }
        if (ok) return {"topo_order", "code_for_topo_order"};
        return {"", ""};
    }
    
    std::pair<std::string, std::string> synthesize_wadj_src_map(
        const std::vector<std::pair<std::pair<WAdjList, int>, std::map<int, double>>>& examples) {
        bool ok = true;
        for (const auto& ex : examples) {
            if (dijkstra(ex.first.first, ex.first.second) != ex.second) { ok = false; break; }
        }
        if (ok) return {"dijkstra", "code_for_dijkstra"};
        return {"", ""};
    }
};

}}}
