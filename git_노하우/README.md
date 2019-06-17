## 외부 프로젝트에 우리의 흔적 남기기 ##



### fork 따기 ###

- 프로젝트로 간다
- fork 버튼 누른다
- 내 remote repository로 프로젝트가 복사된다



### Clone 하기 ###

- 내 remote repository에 있는 git을 clone 한다
- 그러면 내 local repository로 프로젝트가 복사된다.



### Upstream 만들기 ###

- fork된 것은 최신화 되지 않는다.. ㄸ
- 그래서 우리는 원본 (외부 프로젝트 repository)의 최신본이 항상 업데이트 되어야 한다
- 원본 remote repository를 upstream이라 하고  remote 명령어를 이용해 연결해준다

`git remote add upstream 주소(http://~)`



### Fetch 하기 ###

- Upstream 

`git fetch upstream`



### Feature branch 따기 ###

- 기능 추가를 위한 코드 작업은 Feature 브랜치에서 작업한다.
- 이를 위해서는 develop 브랜치에서 **feature/기능 이름** 으로 브랜치를 생성한다.

`git checkout develop`

`git branch feature/기능`

`git checkout feature/기능`

`git add 수정된 파일 이름`

`git commit -m "변경사항"`



### Develop에 반영하기 ###

- Feature 작업이 끝났다면 Develop 브랜치에 merge를 해주어야 한다.
- 이때, Local의 Develop 브랜치 버전을 Upstream의 버전과 맞춰주어야 한다.

`git checkout develop`

`git pull upstream`

- 최신 버전의 Develop 브랜치에서 feature 작업 내용을 merge 해준다

`git merge feature/기능`



### 내 remote repository에 반영하기 ###

- upstream으로 바로 push는 불가능하다(권한이 없어서.)

- upstream으로 pull request를 보내기 전에 내 remote origin에 반영한다.

  `git push origin`



### 흔적 남기기 ###

- 내 remote repository로 가서 변경 사항을 pull request 보낸다!
- 받아주세욧!