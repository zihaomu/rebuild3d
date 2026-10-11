// Continuous triangle-overlap detection. Coordinates are final-atlas texels.
#include "uv_layout.h"

extern "C" int r3d_uv_overlap_mask(int count, const double *coordinates,
                                  int width, int height, uint8_t *affected,
                                  uint64_t *pairs, double *maximum_area) {
    using namespace rebuild3d_uv;
    if (count<1 || width<1 || height<1 || !coordinates || !affected || !pairs || !maximum_area)
        return 1;
    try {
        std::fill(affected,affected+count,0); *pairs=0; *maximum_area=0;
        std::vector<Tri> triangles(count);
        for(int i=0;i<count;i++) for(int k=0;k<3;k++) {
            P p{coordinates[6*i+2*k],coordinates[6*i+2*k+1]};
            if(!std::isfinite(p.x) || !std::isfinite(p.y) || p.x<0 || p.x>width || p.y<0 || p.y>height)
                return 1;
            triangles[i][k]=p;
        }
        Grid grid(width,height,16);
        for(int i=0;i<count;i++) {
            for(int j:grid.candidates(triangles[i])) {
                double area=intersectionArea(triangles[i],triangles[j]);
                if(area>1e-8) {
                    affected[i]=affected[j]=1; ++*pairs;
                    *maximum_area=std::max(*maximum_area,area);
                }
            }
            grid.add(triangles[i],i);
        }
        return 0;
    } catch(const std::exception &) { return 2; }
}
