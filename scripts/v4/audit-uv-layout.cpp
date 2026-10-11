// Offline continuous-UV diagnostic using the production geometry kernel.
#include "../../Runtime/rebuild3d_worker/uv_layout.h"
using namespace rebuild3d_uv;

int main(int argc,char **argv) {
    try {
        if(argc!=12) throw std::runtime_error("triangles labels segments segmentLabels faceCount edgeCount chartCount width height radius report");
        int nf=std::stoi(argv[5]),ne=std::stoi(argv[6]),nc=std::stoi(argv[7]);
        int w=std::stoi(argv[8]),h=std::stoi(argv[9]);double radius=std::stod(argv[10]);
        if(nf<1||ne<1||nc<1||w<1||h<1||radius<1) throw std::runtime_error("Invalid dimensions");
        auto tris=read<Tri>(argv[1],nf);auto labels=read<int32_t>(argv[2],nf);
        auto edges=read<Seg>(argv[3],ne);auto edgeLabels=read<int32_t>(argv[4],ne);
        Grid tg(w,h,16);size_t overlap=0,crossChart=0,overlapLarger=0;double maximum=0;
        std::vector<int> examples;std::vector<bool> affected(nf,false);
        for(int i=0;i<nf;i++) {
            for(int j:tg.candidates(tris[i])) {
                double area=intersectionArea(tris[i],tris[j]);
                if(area>1e-8) {
                    overlap++;crossChart+=labels[i]!=labels[j];maximum=std::max(maximum,area);
                    overlapLarger+=area>1e-6;affected[i]=affected[j]=true;
                    if(examples.size()<32) {examples.push_back(j);examples.push_back(i);}
                }
            }
            tg.add(tris[i],i);
        }
        std::cerr<<"Continuous triangle overlap check completed\n";
        Grid eg(w,h,radius);std::vector<double> nearest(nc,std::numeric_limits<double>::infinity());
        size_t pairs=0;double minSquared=radius*radius;int exampleA=-1,exampleB=-1;
        for(int i=0;i<ne;i++) {
            for(int j:eg.candidates(edges[i],radius)) {
                if(edgeLabels[i]==edgeLabels[j]) continue;
                double d=segmentDistanceSquared(edges[i],edges[j]);
                if(d<=radius*radius) {
                    pairs++;nearest[edgeLabels[i]]=std::min(nearest[edgeLabels[i]],d);
                    nearest[edgeLabels[j]]=std::min(nearest[edgeLabels[j]],d);
                    if(d<minSquared) {minSquared=d;exampleA=j;exampleB=i;}
                }
            }
            eg.add(edges[i],i);
        }
        std::ofstream out(argv[11]);out<<std::setprecision(17);
        out<<"{\"overlappingTrianglePairsAbove1eMinus8TexelSquared\":"<<overlap
           <<",\"pairsAbove1eMinus6TexelSquared\":"<<overlapLarger
           <<",\"crossChartOverlapPairs\":"<<crossChart
           <<",\"affectedFaces\":"<<std::count(affected.begin(),affected.end(),true)
           <<",\"maximumIntersectionTexelSquared\":"<<maximum
           <<",\"exampleOverlappingFacePairsFlat\":[";
        for(size_t i=0;i<examples.size();i++) out<<(i?",":"")<<examples[i];
        out<<"],\"affectedFaceIDs\":[";
        bool first=true;
        for(int i=0;i<nf;i++) if(affected[i]) {out<<(first?"":",")<<i;first=false;}
        out<<"],\"gapSearchRadiusTexels\":"<<radius<<",\"boundaryPairsWithinRadius\":"<<pairs
           <<",\"minimumCrossChartGapTexels\":";
        if(pairs) out<<std::sqrt(minSquared);else out<<"null";
        out<<",\"minimumGapBoundaryPair\":["<<exampleA<<","<<exampleB<<"],\"chartMinimumGapTexels\":[";
        for(int i=0;i<nc;i++) {if(i) out<<",";if(std::isfinite(nearest[i])) out<<std::sqrt(nearest[i]);else out<<"null";}
        out<<"]}\n";
        if(!out) throw std::runtime_error("Could not write report");
    } catch(const std::exception &e) {std::cerr<<e.what()<<"\n";return 1;}
}
