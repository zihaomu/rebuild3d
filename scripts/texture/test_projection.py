"""Checks for errors that can silently produce plausible but misregistered texture."""
import unittest
import numpy as np
from common import camera_project, full_intrinsic, raster, resize_intrinsic


class ProjectionTests(unittest.TestCase):
    def test_full_photo_mapping_includes_padding_and_pixel_centers(self):
        record={'workingWidth':1200,'workingHeight':1600,'uprightWidth':4284,'uprightHeight':5712}
        transform={'workingToModelPixels':[[388/1200,0,65+(388/1200-1)/2],[0,518/1600,(518/1600-1)/2],[0,0,1]]}
        model=np.array([[368.6,0,259],[0,367.6,259],[0,0,1.]])
        full=full_intrinsic(record,model,transform)
        ext=np.c_[np.eye(3),[0,0,0]]
        points=np.array([[.01,.02,.8],[-.2,.1,1],[.2,-.15,.7]])
        px=camera_project(points,ext,full)[:,:2]
        working=(px+.5)*(1600/5712)-.5
        predicted=(np.c_[working,np.ones(3)]@np.array(transform['workingToModelPixels']).T)[:,:2]
        np.testing.assert_allclose(predicted,camera_project(points,ext,model)[:,:2],atol=4e-5)
        np.testing.assert_allclose(resize_intrinsic(full,1600/5712),np.linalg.inv(transform['workingToModelPixels'])@model,atol=1e-10)

    def test_visibility_uses_nearest_surface(self):
        v=np.array([[2,2,2],[10,2,2],[2,10,2],[2,2,1],[10,2,1],[2,10,1]],dtype='float32')
        f=np.array([[0,1,2],[3,4,5]])
        z,ids,b=raster(v,f,14,14)
        self.assertEqual(ids[4,4],1); self.assertAlmostEqual(z[4,4],1)
        self.assertEqual(ids[13,13],-1)
        np.testing.assert_allclose(b[4,4],[.5,.25])

    def test_perspective_interpolation_matches_ray_plane_intersection(self):
        v=np.array([[-.5,-.5,1.],[.7,-.5,2.],[-.5,.8,1.5]])
        k=np.array([[80.,0,50],[0,80,50],[0,0,1]])
        e=np.c_[np.eye(3),[0,0,0]]
        z,ids,b=raster(camera_project(v,e,k),np.array([[0,1,2]]),100,100)
        n=np.cross(v[1]-v[0],v[2]-v[0]); d=n@v[0]
        for x,y in [(35,35),(40,45),(40,55)]:
            self.assertEqual(ids[y,x],0)
            ray=np.linalg.inv(k)@np.array([x,y,1])
            hit=ray*d/(n@ray)
            weights=np.r_[b[y,x],1-b[y,x].sum()]
            np.testing.assert_allclose(weights@v,hit,atol=2e-7)
            self.assertAlmostEqual(z[y,x],hit[2],places=6)


if __name__=='__main__': unittest.main()
