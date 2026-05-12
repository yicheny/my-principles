// 单例模式：保证一个类仅有一个实例，并提供一个访问它的全局访问点

class Singleton {
  private static instance: Singleton;

  private constructor() {}

  static getInstance(): Singleton {
    if (!Singleton.instance) {
      Singleton.instance = new Singleton();
    }
    return Singleton.instance;
  }

  doSomething(): void {
    console.log("Singleton instance is working.");
  }
}

// 演示
const s1 = Singleton.getInstance();
const s2 = Singleton.getInstance();
console.log("s1 === s2:", s1 === s2); // true
