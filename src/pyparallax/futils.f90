!-----------------------------------
! Author: Satoki Tsujino
! Date: 2020/06/06
! Modification: Taiga Tsukada (2025/04/14)
!-----------------------------------
subroutine tri_interp2d(src_x, src_y, src_v, src_priority, dst_x, dst_y, undef, dst_v, dst_priority)
   ! Accelerated triangle interpolation subroutine (main part)
   ! [Note] dst_x, dst_y must be equally spaced
   
   implicit none
   
   double precision, dimension(:,:), intent(in) :: src_x
   double precision, dimension(size(src_x,1),size(src_x,2)), intent(in) :: src_y
   double precision, dimension(size(src_x,1),size(src_x,2)), intent(in) :: src_v ! value at (src_x, src_y)
   double precision, dimension(size(src_x,1),size(src_x,2)), intent(in) :: src_priority ! priority at (src_x,src_y)
   double precision, dimension(:), intent(in) :: dst_x
   double precision, dimension(:), intent(in) :: dst_y
   double precision, intent(in) :: undef
   double precision, dimension(size(dst_x),size(dst_y)), intent(out) :: dst_v ! value at (dst_x, dst_y)
   double precision, dimension(size(dst_x),size(dst_y)), intent(out) :: dst_priority ! priority at (dst_x, dst_y)
   
   integer :: k, l, m, ix, jy, icounter
   integer :: nsi, nti, nxo, nyo, ixmin, ixmax, jymin, jymax, itmp
   integer, dimension(2) :: isqr
   double precision :: intdst_x(size(dst_x)), intdst_y(size(dst_y))
   double precision, dimension(size(src_x,1), size(src_y,2)) :: intsrc_x, intsrc_y
   double precision :: dx, dy, dst_xmin, dst_ymin, dst_v_tmp, dst_priority_tmp
   double precision, dimension(4) :: sqrx, sqry, sqrval, sqrpriority
   double precision, dimension(3) :: interp_x, interp_y, interp_val, interp_priority
   logical :: calc_flag

   logical :: check_triclose
   
   nsi = size(src_x,1)
   nti = size(src_x,2)
   nxo = size(dst_x)
   nyo = size(dst_y)
   
   dx = dst_x(2) - dst_x(1)
   dy = dst_y(2) - dst_y(1)
   dst_xmin = minval(dst_x)
   dst_ymin = minval(dst_y)
   
   dst_v = undef
   dst_priority = undef
   
   !-- Convert dst_x and dst_y to grid point numbers with dst_x(1) = 1 and dst_y(1) = 1
   
   intdst_x = [((dst_x(ix)-dst_xmin)/dx + 1.0d0, ix=1, nxo)]
   intdst_y = [((dst_y(jy)-dst_ymin)/dy + 1.0d0, jy=1, nyo)]
   
   !-- Convert src_x, src_y to real grid point numbers in dst_x, dst_y system
   
   intsrc_x = undef ! src_x, src_y have undef (especially outside the region).
   intsrc_y = undef
   do l = 1, nti
      do k = 1, nsi
         if (src_x(k,l) /= undef .and. src_y(k,l) /= undef) then
            intsrc_x(k,l) = (src_x(k,l)-dst_xmin)/dx + 1.0d0
            intsrc_y(k,l) = (src_y(k,l)-dst_ymin)/dy + 1.0d0
         end if
      end do
   end do
   
   ! Triangulate src_x and src_y starting from the lower left.
   ! (1) check whether the target point (dst_x, dst_y) is included in the triangles,
   ! (2) if so, interpolate linearly in the triangles.
   
   do l = 1, nti-1
      do k = 1, nsi-1
         sqrx = [intsrc_x(k,l), intsrc_x(k+1,l), intsrc_x(k,l+1), intsrc_x(k+1,l+1)]
         sqry = [intsrc_y(k,l), intsrc_y(k+1,l), intsrc_y(k,l+1), intsrc_y(k+1,l+1)]
         sqrval = [src_v(k,l), src_v(k+1,l), src_v(k,l+1), src_v(k+1,l+1)]
         sqrpriority = [src_priority(k,l), src_priority(k+1,l), src_priority(k,l+1), src_priority(k+1,l+1)]
         !-- Verify that all four target neighbors are not undefined.
         calc_flag = .true.
         do m = 1, 4
            if (sqrx(m) == undef) then
               calc_flag = .false.
               exit
            end if
            if (sqry(m) == undef) then
               calc_flag = .false.
               exit
            end if
         end do
         if (calc_flag .eqv. .false.) then ! If no calculation is done, go to the next (k,l) in the cycle.
            cycle
         end if
         !-- To divide a triangle from four adjacent points, diagonals are identified.
         !-- (Judged by the presence or absence of line segment intersections)
         !-- `selopt` to get the longer diagonal point. 
         !-- Since we use triangles with short diagonals, we get the longer diagonal for later processing.
         call check_square_intersect(sqrx, sqry, 'l', isqr)
         !-- Checking if there is a candidate interpolation point in the triangle
         do m = 1, 2 ! There are 2 triangles
            icounter = 1
            do itmp = 1, 4
               if (isqr(m) /= itmp) then ! Construct the m-th triangle off diagonal points
                  interp_x(icounter) = sqrx(itmp)
                  interp_y(icounter) = sqry(itmp)
                  interp_val(icounter) = sqrval(itmp)
                  interp_priority(icounter) = sqrpriority(itmp)
                  icounter = icounter + 1
               end if
            end do
            ixmin = idint(dint(dmin1(interp_x(1),interp_x(2),interp_x(3))))
            ixmax = idint(dint(dmax1(interp_x(1),interp_x(2),interp_x(3)))) + 1
            jymin = idint(dint(dmin1(interp_y(1),interp_y(2),interp_y(3))))
            jymax = idint(dint(dmax1(interp_y(1),interp_y(2),interp_y(3)))) + 1
            if (ixmin > ixmax .or. jymin > jymax) then ! If there is not a candidate interpolation point, cycle
               cycle
            end if
   
            do jy = jymin, jymax ! This `jymin`, `jymax`, `ixmin`, `ixmax` is a point in `dst_v`
               if (jy < 1 .or. jy > nyo) then
                  cycle
               end if
               do ix = ixmin, ixmax
                  if (ix < 1 .or. ix > nxo) then
                     cycle
                  end if
                  if (check_triclose(interp_x, interp_y, [intdst_x(ix),intdst_y(jy)]) .eqv. .false.) then
                     cycle
                  end if
   
                  ! If indeed there is an interpolation point.
                  call tri_interp_1obs(interp_x, interp_y, interp_priority, intdst_x(ix), intdst_y(jy), dst_priority_tmp)
                  if (dst_priority(ix,jy) == undef) then
                     call tri_interp_1obs(interp_x, interp_y, interp_val, intdst_x(ix), intdst_y(jy), dst_v_tmp)
                     dst_v(ix,jy) = dst_v_tmp
                     dst_priority(ix,jy) = dst_priority_tmp
                  else
                     if (dst_priority_tmp > dst_priority(ix,jy)) then ! Preference given to those with higher priority.
                        call tri_interp_1obs(interp_x, interp_y, interp_val, intdst_x(ix), intdst_y(jy), dst_v_tmp)
                        dst_v(ix,jy) = dst_v_tmp
                        dst_priority(ix,jy) = dst_priority_tmp
                     end if
                  end if
               end do
            end do
         end do
      end do
   end do
end subroutine tri_interp2d

subroutine tri_interp_1obs(x, y, val, xp, yp, oval)
   ! Linear interpolation routine in a triangular element.
   ! For the area of the region divided by the interpolation points (xp, yp) in the element,
   ! oval = \sum^{3}_{i}(S_(i)*val(i)) / S, 
   ! S = \sum^{3}_{i}(S_(i))
   ! Each area is obtained from the coordinates of each point and element point of the triangle
   ! using the property that the outer product of the vectors is the area of the parallelogram.

   implicit none
   double precision, intent(in) :: x(3)   ! x-coordinate of each triangle vertex
   double precision, intent(in) :: y(3)   ! y-coordinate of each triangle vertex
   double precision, intent(in) :: val(3) ! Values of each triangle vertex
   double precision, intent(in) :: xp     ! Interpolated x-coordinate in the triangle
   double precision, intent(in) :: yp     ! Interpolated y-coordinate in the triangle
   double precision, intent(inout) :: oval ! Value at interpolation point
   double precision :: Stot, S(3)

   Stot = (x(2)-x(1))*(y(3)-y(1)) - (x(3)-x(1))*(y(2)-y(1))
   S(1) = (x(3)-x(2))*(yp-y(3)) - (xp-x(3))*(y(3)-y(2))
   S(2) = (x(1)-x(3))*(yp-y(1)) - (xp-x(1))*(y(1)-y(3))
   S(3) = (x(2)-x(1))*(yp-y(2)) - (xp-x(2))*(y(2)-y(1))

   oval = (val(1)*S(1)+val(2)*S(2)+val(3)*S(3)) / Stot
end subroutine tri_interp_1obs

subroutine check_square_intersect(x, y, selopt, inum)
   ! -- Find the shorter diagonal from the coordinates of the four points of a quadrangle.
   ! -- If selopt = 'l', the longer diagonal is returned.
   implicit none
   double precision, intent(in) :: x(4) ! 4-point x-coordinate of a quadrangle
   double precision, intent(in) :: y(4) ! 4-point y-coordinate of a quadrangle
   character(1), intent(in) :: selopt 
   integer, intent(out) :: inum(2) ! Coordinates of the shortest (longest) diagonal point (values from 1 to 4)
   logical :: return_long
   logical :: check_intersect

   if (selopt(1:1) == 'l') return_long = .true.
   if (selopt(1:1) == 's') return_long = .false.

   !-- There are only three ways to draw a line connecting two points of a quadrangle (ABCD).
   !-- AB-CD, AC-BD, AD-BC

   if (check_intersect([x(1),x(2)], [y(1),y(2)], [x(3),x(4)], [y(3),y(4)]) .eqv. .true.) then ! AB-CD
      if ((x(1)-x(2))**2+(y(1)-y(2))**2 < (x(3)-x(4))**2+(y(3)-y(4))**2) then ! AB
         if (return_long .eqv. .true.) then
            inum(1:2) = [3,4]
         else
            inum(1:2) = [1,2]
         end if
      else ! CD
         if (return_long .eqv. .true.) then
            inum(1:2) = [1,2]
         else
            inum(1:2) = [3,4]
         end if
      end if
   else
      if (check_intersect([x(1),x(3)], [y(1),y(3)], [x(2),x(4)], [y(2),y(4)]) .eqv. .true.) then ! AC-BD
         if ((x(1)-x(3))**2+(y(1)-y(3))**2 < (x(2)-x(4))**2+(y(2)-y(4))**2) then ! AC
            if (return_long .eqv. .true.) then
               inum(1:2) = [2,4]
            else
               inum(1:2) = [1,3]
            end if
         else ! BD
            if (return_long .eqv. .true.) then
               inum(1:2) = [1,3]
            else
               inum(1:2) = [2,4]
            end if
         end if
      else ! If not above, AD-BC confirmed
         if ((x(1)-x(4))**2+(y(1)-y(4))**2 < (x(2)-x(3))**2+(y(2)-y(3))**2) then ! AC
            if (return_long .eqv. .true.) then
               inum(1:2) = [2,3]
            else
               inum(1:2) = [1,4]
            end if
         else ! BD
            if (return_long .eqv. .true.) then
               inum(1:2) = [1,4]
            else
               inum(1:2) = [2,3]
            end if
         end if
      end if
   end if
end subroutine check_square_intersect

function check_intersect(x1, y1, x2, y2) result(intersection)
   ! Determines the intersection of two line segments.
   ! How to find: Determine the intersection of two line segments defined by 
   ! given {x1,y1} and {x2,y2} from the intersection judgment of the line segments. 

   implicit none

   double precision, intent(in) :: x1(2) ! 1st line x-directional grid point number
   double precision, intent(in) :: y1(2) ! 1st line y-directional grid point number
   double precision, intent(in) :: x2(2) ! 2nd line x-directional grid point number
   double precision, intent(in) :: y2(2) ! 2nd line y-directional grid point number
   double precision :: xa, ya, xb, yb, xc, yc, xd, yd, t1, t2
   logical :: intersection
   intersection = .false.

   xa = x1(1)
   xb = x1(2)
   xc = x2(1)
   xd = x2(2)
   ya = y1(1)
   yb = y1(2)
   yc = y2(1)
   yd = y2(2)

   !-- Determine the intersection of a line passing through {x1,y1} and a line segment {x2,y2}.
   t1 = (xa-xb)*(yc-ya) + (ya-yb)*(xa-xc)
   t2 = (xa-xb)*(yd-ya) + (ya-yb)*(xa-xd)
   if (t1*t2 < 0.0d0) then
      !-- Determine the intersection of a line passing through {x2,y2} and a line segment {x1,y1}.
      t1 = (xc-xd)*(ya-yc) + (yc-yd)*(xc-xa)
      t2 = (xc-xd)*(yb-yc) + (yc-yd)*(xc-xb)
      if (t1*t2 < 0.0d0) then ! If both are not negative, both line segments do not intersect.
         intersection = .true.
         return
      end if
   end if
end function check_intersect

function check_triclose(xposi, yposi, ival) result(triclose)
   ! Checks if the `ival` point lies within the closed curve area enclosed by the triangle.
   ! This judgment is made based on the center of gravity of the triangle, 
   ! regardless of whether the given `xposi` and `yposi` are defined clockwise or counterclockwise.
   ! How to find: The criterion is that the line segment formed by `ival` and the center of gravity 
   ! does not intersect any of the sides of the triangle. The intersection of line segments is 
   ! determined by alternating the intersection of line segments and straight lines.

   implicit none

   double precision, intent(in) :: xposi(3) !  Number of the x grid point at the triangle vertex
   double precision, intent(in) :: yposi(3) ! Number of the y grid point at the triangle vertex
   double precision, intent(in) :: ival(2) ! x- and y-grid point number of the point to be checked
   integer :: ii
   double precision, dimension(4) :: xt, yt
   double precision :: xg, yg, xi, yi, xa, xb, ya, yb, t1, t2

   logical :: triclose
   triclose = .true.

   xt(1:3) = xposi(1:3)
   yt(1:3) = yposi(1:3)
   xt(4) = xposi(1)
   yt(4) = yposi(1)
   xi = ival(1)
   yi = ival(2)

   !-- Calculation of triangle center of mass
   xg = (xt(1)+xt(2)+xt(3)) / 3.0d0
   yg = (yt(1)+yt(2)+yt(3)) / 3.0d0

   !-- For each side, determine the intersection of the center of gravity with the line segment of `ival`.
   do ii = 1,3
      xa = xt(ii)
      xb = xt(ii+1)
      ya = yt(ii)
      yb = yt(ii+1)

      !-- Determines the intersection of a straight line passing through one side of a triangle
      ! and a line segment between the triangle center of mass and the decision point.
      t1 = (xa-xb)*(yg-ya) + (ya-yb)*(xa-xg)
      t2 = (xa-xb)*(yi-ya) + (ya-yb)*(xa-xi)
      if (t1*t2<0.0d0) then
         !-- Intersection judgment of a line passing through the triangle center of gravity
         ! and the judgment point, and a side of the triangle.
         t1 = (xg-xi)*(ya-yg) + (yg-yi)*(xg-xa)
         t2 = (xg-xi)*(yb-yg) + (yg-yi)*(xg-xb)
         if (t1*t2 < 0.0d0) then ! If both are not negative, both line segments do not intersect.
            triclose = .false.
            return
         end if
      end if
   end do
end function check_triclose