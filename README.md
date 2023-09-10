# FGeoDRL
## Cowork
    # 切换回自己的分支、与主分支同步，之后再开发项目
    git checkout <your_branch_name>
    git pull origin main

    # 提交项目更新并push到远程
    git pull origin main
    git add <updated_file_or_path>
    git commit -m "<简单描述更改>"
    git push

    # 合并到主分支
    git checkout main
    git merge <your_branch_name>
    git push
    git checkout <your_branch_name>
    