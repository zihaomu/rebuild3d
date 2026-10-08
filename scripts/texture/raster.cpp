// Small deterministic CPU rasterizer for visibility and UV-surface correspondence.
// Pixel coordinates denote centers; triangles are two-sided. Perspective-correct barycentrics.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
extern "C" void rasterize(int count, const int32_t *faces, const float *xyz,
                         int width, int height, float *depth, int32_t *faceID, float *bary) {
    for (int i=0;i<width*height;i++) {depth[i]=std::numeric_limits<float>::infinity(); faceID[i]=-1;}
    for (int f=0;f<count;f++) {
        const float *a=xyz+3*faces[3*f], *b=xyz+3*faces[3*f+1], *c=xyz+3*faces[3*f+2];
        if (a[2]<=0 || b[2]<=0 || c[2]<=0) continue;
        double den=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1]);
        if (!std::isfinite(den) || std::abs(den)<1e-10) continue;
        int x0=std::max(0,(int)std::ceil(std::min({a[0],b[0],c[0]})));
        int x1=std::min(width-1,(int)std::floor(std::max({a[0],b[0],c[0]})));
        int y0=std::max(0,(int)std::ceil(std::min({a[1],b[1],c[1]})));
        int y1=std::min(height-1,(int)std::floor(std::max({a[1],b[1],c[1]})));
        for(int y=y0;y<=y1;y++) for(int x=x0;x<=x1;x++) {
            double u=((b[1]-c[1])*(x-c[0])+(c[0]-b[0])*(y-c[1]))/den;
            double v=((c[1]-a[1])*(x-c[0])+(a[0]-c[0])*(y-c[1]))/den;
            double w=1-u-v;
            if(std::min({u,v,w}) < -1e-7) continue;
            double inv=u/a[2]+v/b[2]+w/c[2];
            double z=1/inv;
            int p=y*width+x;
            if(z<depth[p]) {depth[p]=z; faceID[p]=f; bary[2*p]=(u/a[2])/inv; bary[2*p+1]=(v/b[2])/inv;}
        }
    }
}
