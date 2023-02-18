---
title: "2019大数据研究中心夏令营上机考试题解"
date: "2020-07-07 18:55:22"
tags: ["Algorithm", "Interview"]
---

*因为就是准备笔试时候自己刷了刷题，所以基本都只在本地跑过了样例*

## 1.最近点对

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 10010
#define ll long long

ll x[maxn], y[maxn];

ll dist(ll x1, ll y1, ll x2, ll y2){
    return (x1 - x2) * (x1 - x2) + (y1 - y2) * (y1 - y2);
}

int main(){
    int t;
    cin>>t;
    while(t--){
        int n;
        ll mindist = 0x7fffffff;
        cin>>n;
        for(int i=0;i<n;++i){
            cin>>x[i]>>y[i];
            for(int j=0;j<i;++j){
                mindist = min(mindist, dist(x[i], y[i], x[j], y[j]));
            }
        }
        cout<<mindist<<endl;
    }
}
```



## 2. 病人排队

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

#define ll long long

struct person{
    string id;
    int age;
    int order;
    person(string id, int age, int order):
    id(id), age(age), order(order){};
};

bool operator< (const person a, const person b){
    if(a.age>=60 && b.age<60){
        return false;
    }
    else if(a.age<60 && b.age>=60){
        return true;
    }
    else if(a.age>=60 && b.age>=60){
        if(a.age != b.age){
            return a.age < b.age;
        }
        else{
            return a.order > b.order;
        }
    }
    else{
        return a.order > b.order;
    }
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int n;
    cin>>n;
    priority_queue<person> q;
    for(int i=0;i<n;++i){
        string id;
        int age;
        cin>>id>>age;
        q.push(person(id, age, i));
    }
    while(!q.empty()){
        auto tmp = q.top();
        q.pop();
        cout<<tmp.id<<endl;
    }
}
```



## 3. 网线主管

`Binary Search`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 10010
#define ll long long

int len[maxn];
int n, k;

bool test(int l){
    int cnt = 0;
    for(int i=0;i<n;++i){
        cnt += len[i] / l;
    }
    return cnt >= k;
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin>>n>>k;
    for(int i=0;i<n;++i){
        double tmp;
        cin>>tmp;
        len[i] = round(tmp * 100);
    }

    if(!test(1)){
        cout<<"0.00"<<endl;
        return 0;
    }

    int l = 1, r = 10000001;
    while(r - l > 1){
        int mid = (l + r) >> 1;
        if(test(mid)){
            l = mid;
        }
        else{
            r = mid;
        }
    }
    printf("%.2lf\n", ((double)l/100));
}
```



## 4. 棋盘问题

`Search`

```c++
#include<bits/stdc++.h>
using namespace std;

int n, k;
int res;
char mat[10][10];
int col[10];

bool isok(int r, int c){
    for(int i=0;i<r;++i){
        if(col[i] == c){
            return false;
        }
    }
    return true;
}

void solve(int r, int rest){
    if(rest == 0){
        ++res;
        return;
    }
    if(r == n){
        return;
    }
    for(int i=0;i<n;++i){
        if(mat[r][i] == '#' && isok(r, i)){
            col[r] = i;
            solve(r+1, rest-1);
        }
    }
    col[r] = -1;
    solve(r+1, rest);
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    while(cin>>n>>k && n!=-1 && k !=-1){
        for(int i=0;i<n;++i){
            cin>>mat[i];
        }
        res = 0;
        solve(0, k);
        cout<<res<<endl;
    }
}
```



## 5. 移动办公

`DP`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 101

int p[maxn], n[maxn];
int dp[2][maxn];

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t, m;
    cin>>t>>m;
    for(int i=0;i<t;++i){
        cin>>p[i]>>n[i];
    }

    dp[0][t] = dp[1][t] = 0;
    for(int i=t-1;i>=0;--i){
        dp[0][i] = max(dp[0][i+1]+p[i], dp[1][i+1]+p[i]-m);
        dp[1][i] = max(dp[1][i+1]+n[i], dp[0][i+1]+n[i]-m);
    }

    cout<<max(dp[0][0], dp[1][0])<<endl;
}
```



## 6. 螺旋加密

`Implementation`

```c++
#include<bits/stdc++.h>
using namespace std;

int row, col;
char mat[24][24];

string getcode(char c){
    if(c == ' '){
        return "00000";
    }
    int n = c - 'A' + 1;
    string res;
    for(int i=0;i<5;++i){
        if(n & 1){
            res.push_back('1');
        }
        else{
            res.push_back('0');
        }
        n >>= 1;
    }
    reverse(res.begin(), res.end());
    return res;
}

void fillblock(const string & code){
    for(int i=0;i<24;++i){
        for(int j=0;j<24;++j){
            mat[i][j] = '0';
        }
    }

    int u = 0, d = row-1, l = 0, r = col-1;
    int i = 0, j = 0;
    int dir[4][2] = {{0, 1}, {1, 0}, {0, -1}, {-1, 0}};
    int dd = 0;
    for(int k=0;k<code.length();++k){
        mat[i][j] = code[k];
        if(j + dir[dd][1] > r){
            ++u;
            dd=1;
        }
        else if(j + dir[dd][1] < l){
            --d;
            dd=3;
        }
        else if(i + dir[dd][0] > d){
            --r;
            dd=2;
        }
        else if(i + dir[dd][0] < u){
            ++l;
            dd=0;
        }
        i += dir[dd][0];
        j += dir[dd][1];
    }
}


int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    cin>>row>>col;
    string s;
    getline(cin, s);

    string mycode = "";
    for(int i=1;i<s.length();++i){
        mycode += getcode(s[i]);
    }

    fillblock(mycode);
    for(int i=0;i<row;++i){
        for(int j=0;j<col;++j){
            cout<<mat[i][j];
        }
    }
    cout<<endl;
}
```



## 7. 单词序列

`BFS`

```c++
#include<bits/stdc++.h>
using namespace std;

int row, col;
char mat[24][24];
map<string, int> dist;

bool diff1(const string& a, const string& b){
    int len = a.length();
    int cnt = 0;
    for(int i=0;i<len;++i){
        if(a[i] != b[i]){
            ++cnt;
        }
    }
    return cnt == 1;
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    string s, e;
    cin>>s>>e;
    dist[s] = 1;
    dist[e] = 42;
    string t;
    while(cin>>t){
        dist[t] = 42;
    }

    queue<string> q;
    q.push(s);
    while(!q.empty()){
        string v = q.front();
        q.pop();
        for(auto tmp: dist){
            if(tmp.second == 42 && diff1(tmp.first, v)){
                dist[tmp.first] = dist[v] + 1;
                q.push(tmp.first);
            }
        }
    }
    if(dist[e] == 42){
        cout<<0<<endl;
    }
    else{
        cout<<dist[e]<<endl;
    }
}
```



## 8. 区间异或和

`Binary Indexed Tree`

```c++
#include<bits/stdc++.h>
using namespace std;

#define maxn 100010

int n, q;
int arr[maxn], nums[maxn];

int lb(int x){
    return x & (-x);
}

void update(int k, int x){
    for(int i=k;i<=n;i+=lb(i))
        arr[i] ^= x;
}

int getsum(int k){
    int res = 0;
    for(int i=k;i>0;i-=lb(i))
        res ^= arr[i];
    return res;
}

int main(){
    while(cin>>n>>q){
        memset(arr, 0, sizeof(arr));
        int res = 0;
        for(int i=1;i<=n;++i){
            update(i, i);
            nums[i] = i;
        }
        for(int i=0;i<q;++i){
            int op, l, r;
            cin>>op>>l>>r;
            if(op){
                update(l, r ^ nums[l]);
                nums[i] = r;
            }
            else{
                res ^= getsum(r) ^ getsum(l-1);
            }
        }
        cout<<res<<endl;
    }
}
```



## 9. 炮兵阵地

`DP`

```c++
#include<bits/stdc++.h>
using namespace std;
int n, m;
char grid[104][10];
int row[104];
int state[104];
int countone[104];
int dp[104][64][64];

int count_one(int k)
{
	int cnt = 0;
	while (k)
	{
		++cnt;
		k = k & (k - 1);
	}
	return cnt;
}

int preprocess()
{
	int cnt = 0;
	for (int i = 0; i < (1 << m); ++i)
	{
		if ((i&(i >> 1)) == 0 && (i&(i >> 2)) == 0)
		{
			countone[cnt] = count_one(i);
			state[cnt++] = i;
		}
	}
	return cnt;
}

int main()
{
	cin >> n >> m;
	int ms = preprocess();
	for (int i = 0; i < n; ++i)
		cin >> grid[i];
	for (int i = 0; i < n; ++i)
		for (int j = 1; j <= m; ++j)
			row[i+1] += (grid[i][j-1] == 'P') << (m - j);
	for (int j = 0; j < ms; ++j)
		if ((state[j] | row[1]) == row[1])
			dp[1][0][j] = countone[j];
	for (int i = 2; i <= n; ++i)
	{
		for (int ppre = 0; ppre < ms; ++ppre)
		{
			if ((state[ppre] | row[i - 2]) != row[i - 2])
				continue;
			for (int pre = 0; pre < ms; ++pre)
			{
				if ((state[pre] | row[i - 1]) != row[i - 1] || (state[ppre] & state[pre]) != 0)
					continue;
				for (int j = 0; j < ms; ++j)
					if ((state[j] | row[i]) == row[i] && (state[j] & state[pre]) == 0 && (state[j] & state[ppre]) == 0)
						dp[i][pre][j] =max(dp[i][pre][j], countone[j]+ dp[i - 1][ppre][pre]);
			}
		}
	}
	int ans = 0;
	for (int i = 0; i < ms; ++i)
		for (int j = 0; j < ms; ++j)
			ans = max(ans,dp[n][i][j]);
	cout << ans << endl;
}
```



## a. 迷宫路口

`DFS`

```c++
#include<bits/stdc++.h>
using namespace std;

int s, n;
int cnt[12];
int mat[32][32];

bool can_place(int x, int y, int size){
    if(x+size<=s && y+size<=s){
        for(int i=0;i<size;++i){
            for(int j=0;j<size;++j){
                if(mat[x+i][y+j] != 0){
                    return false;
                }
            }
        }
        return true;
    }
    else{
        return false;
    }
}

void place(int x, int y, int size){
    for(int i=0;i<size;++i){
        for(int j=0;j<size;++j){
            mat[x+i][y+i] = size;
        }
    }
    --cnt[size];
}

void unplace(int x, int y){
    int size = mat[x][y];
    for(int i=0;i<size;++i){
        for(int j=0;j<size;++j){
            mat[x+i][y+j] = 0;
        }
    }
    ++cnt[size];
}

pair<int, int> next_loc(int x, int y){
    for(int i=x;i<s;++i){
        for(int j=y;j<s;++j){
            if(mat[i][j] == 0){
                return make_pair(i, j);
            }
        }
    }
    return make_pair(-1, -1);
}

bool dfs(int x, int y){
    for(int i=10;i>=1;--i){
        if(cnt[i]){
            if(can_place(x, y, i)){
                place(x, y, i);
                auto loc = next_loc(x, y);
                if(loc.first == -1 && loc.second == -1){
                    return true;
                }
                if(dfs(loc.first, loc.second)){
                    return true;
                }
                unplace(x, y);
            }
        }
    }
    return false;
}

int main(){
    ios::sync_with_stdio(0);
    cin.tie(0);
    cout.tie(0);

    int t;
    cin>>t;
    while(t--){
        cin>>s>>n;
        for(int i=0;i<n;++i){
            int tmp;
            cin>>tmp;
            cnt[tmp]++;
        }
        cout<<(dfs(0, 0)? "YES": "NO")<<endl;
    }
}
```

