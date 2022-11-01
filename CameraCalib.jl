using DelimitedFiles
using Rotations
using LsqFit

function H_Normalize(Pxy)
    m=mean(Pxy[:,1:2],dims=1)                       # 计算均值
    s=std(Pxy[:,1:2],dims=1,corrected=false)        # 计算标准差
    N_p=(Pxy[:,1:2].-m)./s                          # 归一化
    N_p=[N_p ones(size(N_p,1))]                     # 齐次化
    T = [1/s[1] 0 -m[1]/s[1]; 0 1/s[2] -m[2]/s[2]; 0 0 1] # 变换T：Np=T*Pxy
    Inv_T = [s[1] 0 m[1]; 0 s[2] m[2]; 0 0 1]       # 逆变换 IT：Pxy=IT*Np
    return N_p,T,Inv_T
end

function solve_H(Obj_p,Img_p)
    I_N,_,Inv_TI=H_Normalize(Img_p[:,1:2])          # 归一化
    O_N,TO,_=H_Normalize(Obj_p[:,1:2])
    Np=size(O_N,1)
    A=zeros(2*Np,9)
    A[1:2:2*Np,1:3].=O_N                            # 组装
    A[2:2:2*Np,4:6].=O_N
    A[1:2:2*Np,7:9].=-O_N.*I_N[:,1]
    A[2:2:2*Np,7:9].=-O_N.*I_N[:,2]
    F=svd(A,full=false)                             # 求解H
    h=F.Vt[end,:]
    h./=h[end]                                    # 令H33=1
    H_N=reshape(h,(3,3))                           # 按列存，reshape后需要转置
    H=Inv_TI * H_N' * TO
end

function R2Rv(R)
    RR=AngleAxis(R)
    angle=rotation_angle(RR)
    axis=rotation_axis(RR)
    axis.*angle
end

function calib_linear(Obj_p,Img_lis)
    Img_num=size(Img_lis,1)
    H_lis=zeros((Img_num,3,3))
    for (index,Img_p) in enumerate(Img_lis)
        H_lis[index,:,:]=solve_H(Obj_p,Img_p)    # 求解单应
    end
    V=zeros(Img_num*2,6)
    for i = 1:Img_num
        H=transpose(H_lis[i,:,:])                # 与文章保持一致
        v11=[H[1,1]^2, 2*H[1,2]*H[1,1], H[1,2]^2, 2*H[1,1]*H[1,3], 2*H[1,2]*H[1,3], H[1,3]^2]
        v12=[H[1,1]*H[2,1], H[1,2]*H[2,1]+H[1,1]*H[2,2], H[1,2]*H[2,2], H[1,1]*H[2,3]+H[1,3]*H[2,1], H[1,2]*H[2,3]+H[1,3]*H[2,2], H[1,3]*H[2,3]]
        v22=[H[2,1]^2, 2*H[2,2]*H[2,1], H[2,2]^2, 2*H[2,1]*H[2,3], 2*H[2,2]*H[2,3], H[2,3]^2]
        V[i*2-1,:].=v12                          # 组装V
        V[i*2,:].=v11-v22
    end
    F=svd(V,full=false)
    Bv=F.Vt[end,:]                               # 求解Bv
    Bv./=Bv[end]                                 
    v0=(Bv[2]*Bv[4]-Bv[1]*Bv[5])/(Bv[1]*Bv[3]-Bv[2]*Bv[2])
    lam=Bv[6]-(Bv[4]*Bv[4]+v0*(Bv[2]*Bv[4]-Bv[1]*Bv[5]))/Bv[1]
    fx=sqrt(lam/Bv[1])
    fy=sqrt(lam*Bv[1]/(Bv[1]*Bv[3]-Bv[2]*Bv[2]))
    s=-Bv[2]*fx*fx*fy/lam
    u0=s*v0/fy-Bv[4]*fx*fx/lam
    K=[fx s u0;0 fy v0;0 0 1]                    # 重构K
    RT_lis=zeros(6,Img_num)
    for i=1:Img_num
        H=H_lis[i,:,:]
        r12t = K\H
        r12t ./= norm(r12t[:,1])
        R=[r12t[:,1:2] cross(r12t[:,1],r12t[:,2])]
        RT_lis[1:3,i] .= R2Rv(R)
        RT_lis[4:6,i] .= r12t[:,3]
    end
    camera_p = [fx,fy,s,u0,v0,0,0]               # 2个0为畸变参数
    pose_p = vec(RT_lis)
    [camera_p;pose_p]
end

function project(obj_p,camera_p,pose_p)
    # x 为obj_p
    R=RotationVec(pose_p[1],pose_p[2],pose_p[3])      # 构建旋转矩阵
    points_proj = obj_p*R' .+ pose_p[4:6]'            # 点为行向量，需要转置R
    points_proj = points_proj[:,1:2]./points_proj[:,3:3] #齐次坐标

    fx,fy = camera_p[1:2]
    s = camera_p[3]
    u0,v0 = camera_p[4:6]
    k1,k2 = camera_p[6:7]

    n = sum(points_proj.^2,dims=2)                    # 添加畸变
    r = 1 .+ k1.*n + k2.* n.^2
    points_proj .*= r
    res = points_proj .* [fx fy] .+ [u0 v0]
    res[:,1] += points_proj[:,2] .* s 
    res
end

function proj_model(x,p)
    camera_p=p[1:7]                               # 提取参数
    pose_p=reshape(p[8:end],6,:)
    pose_num=size(pose_p,2)
    res=zeros(pose_num*size(x,1),2)
    for i=1:pose_num
        res[1+(i-1)*size(x,1):i*size(x,1),:] .= project(x,camera_p,pose_p[:,i])
    end
    vec(res)
end

function calib_nonlinear(Obj_p,Img_lis,p0)
    y_data = vec(vcat(Img_lis...)[:,1:2])
    curve_fit(proj_model,Obj_p,y_data,p0; autodiff=:finiteforward)
end

obj_p=readdlm("test/TEST_obj_54_3.txt")
obj_p_num=size(obj_p,1)
img_p=readdlm("test/TEST_img_13_54_2.txt")
img_lis=[[img_p[1+(i-1)*obj_p_num:i*obj_p_num,:] ones(obj_p_num)] for i in 1:div(size(img_p,1),obj_p_num)]

p0=calib_linear(obj_p,img_lis)
fit=calib_nonlinear(obj_p,img_lis,p0)
camera_p = fit.param[1:7]
@show camera_p
RMSE=sqrt(mean(fit.resid.^2))
@show RMSE