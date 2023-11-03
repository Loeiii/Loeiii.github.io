# Linux学习笔记

## 常用指令

### 文件夹操作

#### 查看路径

`pwd`：将当前目录的全物理路径名称（从根目录）写入标准输出

​	**可选参数:**

		- -L：当前工作目录的逻辑路径
		- -P：显示当前目录的绝对路径

#### 压缩文件夹

```
zip -r 压缩后的文件名.zip 要压缩的文件夹名 # -r 表示对目录进行递归
```

#### SCP文件传输

使用SCP进行文件传输需要先建立SSH连接，随后对需要传输的文件使用下列指令即可完成文件的传输

```
scp (-r)(-C) file dest_username@deat_ip:dest_path
```

其中，`file`为要传输的文件，`dest_username`为用户在目标服务器的用户名，`dest_ip`为目标服务器IP地址，`dest_path`为目标文件夹，传输后的文件放在该文件夹下

**注：**如果上传目录不存在，scp会试图创建它。如果想覆盖已存在的文件,可以使用`-C`参数，**添加`-r`参数**可以递归上传整个目录

#### 删除文件夹

```
rm -rf filename
```

## CentOS

### 创建用户及设置密码

#### 创建用户

在root权限下，使用命令`useradd -m username`即可完成用户的创建

#### 设置密码

在完成用户创建之后，使用`passwd username`命令进入到密码设置，输入密码后需要再次输入即可完成密码设置

#### 进入账户

使用命令`su - username`即可从根目录下进入自己账户或者切换到相应的用户，同时也可以使用`sudo su - username`从其他账户切换为自己的账户

#### 退出账户

使用命令`logout`即可退出用户，返回`root`

### SSH远程连接

#### 创建公钥、私钥

```
# 在本地终端中运行
ssh-keygen -t rsa
```

#### 获取公钥

```
# 在本地终端中运行命令并复制输出的公钥
cat ~/.ssh/id_rsa.pub
```

#### 创建新目录及文件

如果最终用户的主目录中没有`.ssh`文件的话，需要在服务器端创建相应文件夹

```
mkdir -p /home/user_name/.ssh
```

随后在`.ssh`文件夹中创建`authorized_keys`文件

```
touch authorized_keys
```

使用文本编译器（如`vim`）打开`authorized_keys`并将复制的公钥粘贴入文件中

```
vim authorized_keys
```

#### 更改文件权限

如果使用上述办法后仍不能ssh远程连接，使用下列方法修改文件权限

```
chmod 700 .ssh 
chmod 600 .ssh/authorized_keys
```

如果我们是为其他用户创建的上述文件，需要使用以下命令更改文件的用户所有权

```
chown -R username:username /home/username/.ssh
```

#### 连接

完成密钥拷贝及文件权限更改后，即可与服务器建立连接

```
ssh remote_username@remote_ip
```

其中，`remote_uername`和`remote_ip`分别是用户在服务器上的用户名和服务器IP地址

### ~~使用跳板机~~

#### ~~跳板机转发端口打开Jupyter~~

~~用端口转发可以打开jupyter~~

~~ssh -L 127.0.0.1:80:127.0.0.1:80 [jupyter-yhs@10.102.2.178~~
