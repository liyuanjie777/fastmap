/*
 * Copyright (c) 2026 Yuanjie Li. All rights reserved.
 * Author: Yuanjie Li
 * See LICENSE in the project root for terms.
 */

#include "graph_decoder.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {
constexpr double NI = -std::numeric_limits<double>::infinity();
thread_local std::string error_message;
struct NoPath : std::runtime_error { using std::runtime_error::runtime_error; };
double ladd(double a, double b) {
    if (a == NI) return b;
    if (b == NI) return a;
    double hi = std::max(a,b),lo=std::min(a,b);
    return hi + std::log1p(std::exp(lo-hi));
}
bool legal(float x) { return !std::isnan(x) && x != std::numeric_limits<float>::infinity(); }
float prob(double x) { return static_cast<float>(std::exp(std::min(0.0,x))); }
size_t mul(size_t a, size_t b) {
    if(b && a > std::numeric_limits<size_t>::max() / b) throw std::invalid_argument("array size overflow");
    return a * b;
}
struct Node {
    int32_t state;
    size_t prefix;
    double mass, best, rank;
    size_t parent;
    int32_t edge;
};
using Key=std::pair<size_t,int32_t>;
struct KeyHash {
    size_t operator()(const Key& x) const {
        return std::hash<size_t>{}(x.first) ^ (std::hash<int32_t>{}(x.second)+0x9e3779b9+(x.first<<6)+(x.first>>2));
    }
};
void prune(std::vector<Node>& v,int32_t width,double cut) {
    std::sort(v.begin(),v.end(),[](const Node&a,const Node&b) {
        if(a.rank!=b.rank) return a.rank>b.rank;
        if(a.prefix!=b.prefix) return a.prefix<b.prefix;
        return a.state<b.state;
    });
    if(v.empty()) throw NoPath("no candidate reaches an allowed final state");
    double threshold=v.front().rank-cut;
    size_t keep=0;
    while(keep<v.size() && keep<static_cast<size_t>(width) && v[keep].rank>=threshold) ++keep;
    v.resize(keep);
}
}

extern "C" const char* graph_decoder_last_error() { return error_message.c_str(); }

extern "C" int graph_beam_decode(
    const float* scores,const int32_t* lengths,int32_t B,int32_t T,int32_t S,int32_t E,
    const int32_t* src,const int32_t* dst,const int32_t* labels,
    const float* start,const float* end,int32_t beam_width,double beam_cut,
    int32_t* tokens,int32_t* token_times,float* token_conf,int32_t* out_lengths,
    int32_t* states,float* state_conf,int32_t* path_edges,float* edge_conf,
    double* log_z,double* sequence_log_mass,double* path_log_score) {
    try {
        error_message.clear();
        if(B<1 || T<0 || T==std::numeric_limits<int32_t>::max() || S<1 || E<1 || beam_width<1 || std::isnan(beam_cut) || beam_cut<0)
            throw std::invalid_argument("invalid dimensions, beam_width, or beam_cut");
        if(!scores || !lengths || !src || !dst || !labels || !tokens || !token_times || !token_conf || !out_lengths || !states || !state_conf || !path_edges || !edge_conf || !log_z || !sequence_log_mass || !path_log_score)
            throw std::invalid_argument("null required pointer");
        size_t BT=mul(B,T),BS=mul(B,static_cast<size_t>(T)+1);
        mul(BT,E); mul(static_cast<size_t>(T)+1,S);
        std::vector<std::vector<int32_t>> outgoing(S);
        for(int32_t e=0;e<E;++e) {
            if(src[e]<0 || src[e]>=S || dst[e]<0 || dst[e]>=S || labels[e]<-1)
                throw std::invalid_argument("invalid edge endpoint or label");
            outgoing[src[e]].push_back(e);
        }
        for(int32_t s=0;s<S;++s)
            if((start && !legal(start[s])) || (end && !legal(end[s])))
                throw std::invalid_argument("NaN/+Inf in start/end potentials");
        std::fill_n(tokens,BT,-1); std::fill_n(token_times,BT,-1);
        std::fill_n(token_conf,BT,0); std::fill_n(path_edges,BT,-1); std::fill_n(edge_conf,BT,0);
        std::fill_n(states,BS,-1); std::fill_n(state_conf,BS,0);

        for(int32_t b=0;b<B;++b) {
            int32_t N=lengths[b];
            if(N<0 || N>T) throw std::invalid_argument("lengths must be in [0,T]");
            const float* x=scores+static_cast<size_t>(b)*T*E;
            for(size_t i=0;i<static_cast<size_t>(N)*E;++i)
                if(!legal(x[i])) throw std::invalid_argument("NaN/+Inf in valid score frames");
            // a[t,s]/back[t,s]: state AFTER t consumed frames. t=0 is initial.
            std::vector<double> a((static_cast<size_t>(N)+1)*S,NI),back(a.size(),NI);
            for(int32_t s=0;s<S;++s) {
                a[s]=start?start[s]:0.0;
                back[static_cast<size_t>(N)*S+s]=end?end[s]:0.0;
            }
            for(int32_t t=0;t<N;++t)
                for(int32_t e=0;e<E;++e) {
                    size_t dest=(static_cast<size_t>(t)+1)*S+dst[e];
                    a[dest]=ladd(a[dest],a[static_cast<size_t>(t)*S+src[e]]+x[static_cast<size_t>(t)*E+e]);
                }
            for(int32_t t=N;t-->0;)
                for(int32_t e=0;e<E;++e) {
                    size_t origin=static_cast<size_t>(t)*S+src[e];
                    back[origin]=ladd(back[origin],x[static_cast<size_t>(t)*E+e]+back[(static_cast<size_t>(t)+1)*S+dst[e]]);
                }
            double z=NI;
            for(int32_t s=0;s<S;++s) z=ladd(z,a[static_cast<size_t>(N)*S+s]+(end?end[s]:0.0));
            if(z==NI) throw NoPath("full graph has no finite path for read "+std::to_string(b));
            log_z[b]=z;

            // Intern output prefixes exactly. Hash collisions cannot merge distinct keys.
            std::unordered_map<Key,size_t,KeyHash> prefix_children;
            size_t next_prefix=1; // 0 denotes empty output.
            std::vector<std::vector<Node>> history(static_cast<size_t>(N)+1);
            for(int32_t s=0;s<S;++s) {
                double v=start?start[s]:0.0;
                if(v!=NI && back[s]!=NI) history[0].push_back({s,0,v,v,v+back[s],0,-1});
            }
            prune(history[0],beam_width,beam_cut);
            for(int32_t t=0;t<N;++t) {
                auto& prev=history[t]; auto& cur=history[t+1];
                std::unordered_map<Key,size_t,KeyHash> merge;
                for(size_t parent=0;parent<prev.size();++parent) {
                    const auto& node=prev[parent];
                    for(int32_t e:outgoing[node.state]) {
                        double w=x[static_cast<size_t>(t)*E+e];
                        double guide=back[(static_cast<size_t>(t)+1)*S+dst[e]];
                        if(w==NI || guide==NI) continue;
                        size_t prefix=node.prefix;
                        if(labels[e]>=0) {
                            auto [it,inserted]=prefix_children.emplace(Key{prefix,labels[e]},next_prefix);
                            if(inserted) ++next_prefix;
                            prefix=it->second;
                        }
                        Key key{prefix,dst[e]};
                        auto [it,inserted]=merge.emplace(key,cur.size());
                        double mass=node.mass+w, best=node.best+w;
                        if(inserted) cur.push_back({dst[e],prefix,mass,best,mass+guide,parent,e});
                        else {
                            auto& old=cur[it->second];
                            old.mass=ladd(old.mass,mass);
                            old.rank=old.mass+guide;
                            if(best>old.best) { old.best=best; old.parent=parent; old.edge=e; }
                        }
                    }
                }
                prune(cur,beam_width,beam_cut);
            }
            struct Final { double mass=NI,best=NI; size_t index=0; };
            std::map<size_t,Final> finals;
            for(size_t j=0;j<history[N].size();++j) {
                const auto& node=history[N][j];
                double finish=end?end[node.state]:0.0;
                if(finish==NI) continue;
                auto& f=finals[node.prefix];
                f.mass=ladd(f.mass,node.mass+finish);
                if(node.best+finish>f.best) { f.best=node.best+finish; f.index=j; }
            }
            if(finals.empty()) throw NoPath("beam retained no complete path");
            Final chosen; bool found=false;
            for(const auto& item:finals) if(!found || item.second.mass>chosen.mass) {chosen=item.second;found=true;}
            sequence_log_mass[b]=chosen.mass; path_log_score[b]=chosen.best;
            size_t j=chosen.index, frame_off=static_cast<size_t>(b)*T, state_off=static_cast<size_t>(b)*(T+1);
            for(int32_t t=N;t>0;--t) {
                const auto& node=history[t][j];
                states[state_off+t]=node.state;
                path_edges[frame_off+t-1]=node.edge;
                j=node.parent;
            }
            states[state_off]=history[0][j].state;
            for(int32_t t=0;t<=N;++t) {
                int32_t s=states[state_off+t];
                state_conf[state_off+t]=prob(a[static_cast<size_t>(t)*S+s]+back[static_cast<size_t>(t)*S+s]-z);
            }
            int32_t count=0;
            for(int32_t t=0;t<N;++t) {
                int32_t e=path_edges[frame_off+t];
                auto log_edge=[&](int32_t f) {
                    return a[static_cast<size_t>(t)*S+src[f]]+x[static_cast<size_t>(t)*E+f]+back[(static_cast<size_t>(t)+1)*S+dst[f]]-z;
                };
                edge_conf[frame_off+t]=prob(log_edge(e));
                if(labels[e]>=0) {
                    double label_mass=NI;
                    for(int32_t f=0;f<E;++f) if(labels[f]==labels[e]) label_mass=ladd(label_mass,log_edge(f));
                    tokens[frame_off+count]=labels[e];
                    token_times[frame_off+count]=t;
                    token_conf[frame_off+count]=prob(label_mass);
                    ++count;
                }
            }
            out_lengths[b]=count;
        }
        return 0;
    } catch(const NoPath& e) { error_message=e.what(); return 2; }
      catch(const std::exception& e) { error_message=e.what(); return 1; }
      catch(...) { error_message="unknown decoder failure"; return 1; }
}
