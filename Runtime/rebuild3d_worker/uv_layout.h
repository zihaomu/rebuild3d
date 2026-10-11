#pragma once
// Offline continuous-UV diagnostic. Does not change the production atlas.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace rebuild3d_uv {
struct P { double x, y; };
P operator-(P a, P b) { return {a.x-b.x,a.y-b.y}; }
P operator+(P a, P b) { return {a.x+b.x,a.y+b.y}; }
P operator*(P a, double b) { return {a.x*b,a.y*b}; }
double cross(P a, P b) { return a.x*b.y-a.y*b.x; }
double dot(P a, P b) { return a.x*b.x+a.y*b.y; }
using Tri = std::array<P,3>;
using Seg = std::array<P,2>;

template<class T> std::vector<T> read(const char *path, size_t count) {
    std::ifstream f(path,std::ios::binary);
    std::vector<T> v(count);
    if (!f.read(reinterpret_cast<char*>(v.data()), count*sizeof(T)) || f.peek()!=EOF)
        throw std::runtime_error("Invalid binary input length");
    return v;
}

double intersectionArea(Tri a, Tri b) {
    if (cross(b[1]-b[0],b[2]-b[0])<0) std::swap(b[1],b[2]);
    std::vector<P> polygon(a.begin(),a.end());
    for (int e=0;e<3 && !polygon.empty();e++) {
        std::vector<P> next;
        P start=b[e], edge=b[(e+1)%3]-start;
        for (size_t k=0;k<polygon.size();k++) {
            P p=polygon[k], q=polygon[(k+1)%polygon.size()];
            double dp=cross(edge,p-start), dq=cross(edge,q-start);
            if (dp>=0) next.push_back(p);
            if ((dp>=0)!=(dq>=0)) next.push_back(p+(q-p)*(dp/(dp-dq)));
        }
        polygon=std::move(next);
    }
    if (polygon.size()<3) return 0;
    double twice=0;
    for(size_t k=1;k+1<polygon.size();k++)
        twice+=cross(polygon[k]-polygon[0],polygon[k+1]-polygon[0]);
    return std::abs(twice)*.5;
}

double pointSegment(P p, Seg s) {
    P d=s[1]-s[0];
    double l=dot(d,d), t=l>0 ? std::clamp(dot(p-s[0],d)/l,0.0,1.0) : 0;
    P delta=p-(s[0]+d*t);
    return dot(delta,delta);
}
double segmentDistanceSquared(Seg a, Seg b) {
    P r=a[1]-a[0], s=b[1]-b[0]; double den=cross(r,s);
    if (den!=0) {
        double t=cross(b[0]-a[0],s)/den, u=cross(b[0]-a[0],r)/den;
        if(t>=0 && t<=1 && u>=0 && u<=1) return 0;
    }
    return std::min({pointSegment(a[0],b),pointSegment(a[1],b),
                     pointSegment(b[0],a),pointSegment(b[1],a)});
}

struct Grid {
    int nx,ny; double cell;
    std::vector<std::vector<int>> bins;
    Grid(int w,int h,double s):nx(int(std::ceil(w/s))+1),ny(int(std::ceil(h/s))+1),cell(s),bins(nx*ny) {}
    template<class Shape> std::array<int,4> bounds(Shape s, double pad=0) {
        double x0=s[0].x,x1=x0,y0=s[0].y,y1=y0;
        for(P p:s) { x0=std::min(x0,p.x);x1=std::max(x1,p.x);y0=std::min(y0,p.y);y1=std::max(y1,p.y); }
        return {std::clamp(int(std::floor((x0-pad)/cell)),0,nx-1),
                std::clamp(int(std::floor((x1+pad)/cell)),0,nx-1),
                std::clamp(int(std::floor((y0-pad)/cell)),0,ny-1),
                std::clamp(int(std::floor((y1+pad)/cell)),0,ny-1)};
    }
    template<class Shape> std::vector<int> candidates(Shape s,double pad=0) {
        auto b=bounds(s,pad);std::vector<int> result;
        for(int y=b[2];y<=b[3];y++) for(int x=b[0];x<=b[1];x++) {
            auto &v=bins[y*nx+x];result.insert(result.end(),v.begin(),v.end());
        }
        std::sort(result.begin(),result.end());
        result.erase(std::unique(result.begin(),result.end()),result.end());
        return result;
    }
    template<class Shape> void add(Shape s,int id) {
        auto b=bounds(s);
        for(int y=b[2];y<=b[3];y++) for(int x=b[0];x<=b[1];x++) bins[y*nx+x].push_back(id);
    }
};

}
